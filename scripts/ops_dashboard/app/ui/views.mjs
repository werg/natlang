// The dashboard's own views (Now, Rounds, Data flow, Held-out, Machines, Units, Teachers, Open items, More). Each view reads the exporter's snapshot
// (S.snap, with the expert-iteration section in snap.ei) and draws it; nothing here fetches. `ctx` is the page shell (app.mjs).
import { num, fmt, gb, pct, signed, dur, ago, clock, dayClock, shortUnit, compact, rate, etaText, where, roundSteps, roundEta, roundSegments, hourlyBars, progress, activity, outcomeParts, matches, memorySegments, latestHeld } from './model.mjs';
import { h, lineChart, spark, segBar, slotVar, barStack, gantt, ladder } from './charts.mjs';
import { link as dl } from './data.mjs';

const SOURCES = {   // fixed colour slot per training-data source: colour follows the entity
  'rl-own': ['Own verified successes', 2], filled: ['Teacher data, student-filled reasoning', 0], teacher: ['Teacher data', 3], 'jolt-guided': ['Guided successes', 4],
  'neuralese-graph': ['Neuralese graphs', 6], operators: ['Operators', 5], live: ['Live corpus', 7], fallow22: ['Broad graphs', 1] };
const STAGES = { prepare: { label: 'Frontier, data, build', slot: 0 }, sft: { label: 'SFT', slot: 2 }, publish: { label: 'Merge and publish', slot: 3 } };
const HOSTS = [{ key: 'dgx', label: 'DGX', slot: 0 }, { key: 'colab', label: 'Colab A100', slot: 2 }, { key: 'pop', label: 'Pop', slot: 3 }];
const TIERS = [['solved', 'solved', 2], ['sometimes', 'sometimes solved', 3], ['never', 'never solved', 7], ['unmeasured', 'unmeasured', 1]];

export function makeViews(ctx) {
  const { S, go, askBtn, alertsView, now, chartPanel, machineChip, meter, section, stat, runLink } = ctx;
  const snap = () => S.snap, ei = () => S.snap?.ei;
  const chip = (text, tone = '', title) => h('span', { class: `chip ${tone}`, title }, text);
  const tile = (value, label, tone = '') => h('div', { class: `tile ${tone}` }, h('b', {}, value), h('span', {}, label));
  const loading = what => h('div', { class: 'card muted' }, snap() ? what : 'Loading…');
  const card = (title, right, ...kids) => h('div', { class: 'card' }, h('header', {}, h('h2', {}, title), right || null), ...kids);
  const more = (label, route) => h('button', { class: 'btn sm', type: 'button', onclick: () => go(route) }, label);
  const latestRound = state => (ei()?.rounds || []).filter(r => r.state === state).at(-1);
  const heldLine = (g) => g ? `${g.candidate} measured ${g.candidate_result ? `${rate(g.candidate_result.rate)} (${g.candidate_result.scored}/${g.candidate_result.ran} scored)` : 'not yet'}` : '';

  /* ---------- round pipeline strip ---------- */
  function stepsStrip(r) {
    const steps = roundSteps(r, ei()?.held, now());
    return h('ol', { class: 'steps', 'aria-label': `Steps of round ${r.id}` }, steps.map(s => h('li', { class: s.state, 'aria-current': s.state === 'active' ? 'step' : null },
      h('i', { 'aria-hidden': 'true' }, s.state === 'done' ? '✓' : s.state === 'active' ? '●' : ''), h('b', {}, s.label), h('span', {}, s.detail || ' '))));
  }
  function roundCard(r, { compactCard = false } = {}) {
    const e = roundEta(r), s = r.sft, size = r.size || {};
    const stateChip = r.state === 'sft' ? chip(`SFT on ${where(/g4b/.test(r.host || '') ? 'colab-g4b' : /colab/.test(r.host || '') ? 'colab' : 'dgx')}`, 'ok') : r.state === 'preparing' ? chip('preparing', 'open') : r.state === 'ready' ? chip('built, waiting for the GPU', 'open') : chip(`published v${r.version}`, 'ok');
    const el = h('div', { class: `card round ${r.state}` }, h('header', {}, h('h2', {}, `Round ${r.id}`, h('span', { class: 'muted small' }, r.trained_from != null ? ` · v${r.trained_from} → v${r.trained_from + 1}` : '')), stateChip), stepsStrip(r));
    if (r.state === 'sft' && s) {
      el.append(h('div', { class: 'prog' }, meter(s.step / s.of, 'good'), h('div', { class: 'row between small' }, h('span', {}, `step ${s.step} of ${s.of} (${Math.round(100 * s.step / s.of)}%)`), h('span', { class: 'muted' }, `${s.s_per_step ? s.s_per_step + ' s/step' : ''}`))),
        h('div', { class: 'eta' }, h('b', {}, `SFT ends ${etaText(e.sftEnd, now())}`), h('span', { class: 'muted' }, `, then merge and publish ${e.publish ? `≈ ${clock(e.publish)}` : ''}. The log is read from the learner about every 20 minutes (last step ${ago(s.step_ts, now())}).`)));
    } else if (r.state === 'preparing') {
      const t = ei()?.teacher;
      el.append(h('div', { class: 'small muted' }, `Prepared in parallel with the running SFT. Teacher campaign ${t ? `${t.tasks_done}/${t.eligible ?? '?'} tasks, ${t.success} verified successes` : 'not started'}${r.data_progress != null ? `; student fill ${r.data_progress} episodes` : ''}.`));
    } else if (r.state === 'ready') el.append(h('div', { class: 'small muted' }, 'Built; waits for the learner to be free.'));
    if (!compactCard) el.append(roundFacts(r));
    return el;
  }
  const roundFacts = r => {
    const size = r.size || {}, mix = r.mix;
    return h('div', { class: 'kv small', style: 'margin-top:8px' }, size.max_target_tokens ? [h('span', {}, 'size'), h('span', {}, `${compact(size.max_target_tokens)} target tokens, ${size.hours} h on ${/A100/.test(size.gpu || '') ? 'A100' : size.gpu || size.placement}`)] : null,
      mix ? [h('span', {}, 'built'), h('span', {}, `${compact(mix.chains)} chains, ${compact(mix.forward_tokens)} forward tokens`)] : null,
      r.episodes_in ? [h('span', {}, 'own episodes'), h('span', {}, compact(r.episodes_in))] : null);
  };

  /* ---------- Now ---------- */
  function statusCard(hv, e) {
    const train = latestRound('sft'), prep = latestRound('preparing');
    const prog = progress(e), actors = e.collection?.actors || [];
    const perHour = actors.reduce((a, x) => a + (x.rate_h || 0), 0);
    const bits = [];
    if (train?.sft) bits.push(`${train.id} is training: SFT ${Math.round(100 * train.sft.step / train.sft.of)}%, published ${etaText(roundEta(train).publish, now())}.`);
    else if (e.phase?.phase) bits.push(`Phase: ${e.phase.phase}.`);
    if (prep) bits.push(`${prep.id} is being prepared.`);
    const el = h('div', { class: `card hero ${hv.level}` }, h('div', { class: 'q' }, `natlang training · ${e.run}`),
      h('div', { class: `big tone-${hv.level}` }, h('span', { 'aria-hidden': 'true' }, hv.level === 'ok' ? '✔' : hv.level === 'crit' ? '✖' : '▲'), hv.open.length ? `${hv.open.length} thing${hv.open.length === 1 ? '' : 's'} to look at` : 'All running'),
      h('div', { class: 'sub' }, bits.join(' ') || 'No round is active.'),
      h('div', { class: 'tiles' }, tile(`v${e.phase?.latest_version ?? '?'}`, `newest policy${e.lineage?.at(-1) ? '' : ''}`), tile(train ? `${Math.round(100 * (train.sft?.step || 0) / (train.sft?.of || 1))}%` : '—', train ? `${train.id} SFT` : 'no SFT'),
        tile(`${compact(perHour)}/h`, 'fresh episodes'), tile(prog.tone === 'flat' && !prog.headline.includes('pts') ? prog.headline : prog.headline, prog.tone === 'flat' ? 'held pass rate' : 'held vs reference', prog.tone)));
    return el;
  }
  function progressCard(e) {
    const held = e.held, prog = progress(e);
    const pts = (held?.results || []).map(r => ({ label: r.label.replace(/^v(\d+) \(.*\)$/, 'v$1 ref'), y: r.rate, lo: r.task_se != null ? r.rate - 1.96 * r.task_se : null, hi: r.task_se != null ? r.rate + 1.96 * r.task_se : null, series: r.series, scored: r.scored, ran: r.ran }));
    const g = held?.gate, p = (held?.in_progress || [])[0];
    const el = card('Is training making progress?', more('Held-out details', 'held'),
      h('div', { class: `verdict tone-${prog.tone}` }, h('b', {}, prog.headline), h('span', {}, ` ${prog.sub}`)), h('div', { class: 'small muted' }, prog.detail));
    if (pts.length) el.append(ladder(pts.slice(-7)));
    else el.append(h('div', { class: 'muted' }, 'No held-out result yet.'));
    const lines = [];
    if (g?.failing) lines.push(`The verdict has failed ${g.failing.count} times (${g.failing.error}).`);
    if (p) lines.push(`${p.label} is re-measured on Pop: ${p.done}/${p.of} episodes, about ${dur(p.eta_s)} left.`);
    if (lines.length) el.append(h('div', { class: 'small', style: 'margin-top:6px' }, lines.join(' ')));
    return el;
  }
  function freshDataCard(e) {
    const c = e.collection, t = snap().teachers, actors = c?.actors || [];
    const el = card('How much fresh data is flowing?', more('Data flow', 'flow'));
    if (!c) return el.append(h('div', { class: 'muted' }, 'No collection yet.')), el;
    const bars = hourlyBars(c.hourly, now(), 12), total = bars.at(-1);
    el.append(barStack(bars, { keys: HOSTS, height: 130 }));
    el.append(h('div', { class: 'rows small' }, actors.map(a => h('div', { class: 'mini' }, h('span', { class: 't' }, `${where(a.host === 'colab' ? 'colab' : a.host)} · v${a.version}`), h('span', { class: 'r' }, `${a.rate_h}/h · ${a.pass_rate_h != null ? rate(a.pass_rate_h) : '—'} pass`))),
      !actors.length ? h('div', { class: 'muted' }, 'No collector is running.') : null,
      t ? h('div', { class: 'mini' }, h('span', { class: 't' }, `Teachers · ${t.pool.replace('teacher-', '')}`), h('span', { class: 'r' }, `${t.success} verified from ${t.tasks_done}/${t.eligible ?? '?'} tasks`)) : null));
    return el;
  }
  function machineMini(key, title, e) {
    const m = snap().machines?.[key === 'dgx' ? 'dgx' : key === 'pop' ? 'pop' : null];
    const gpu = key === 'dgx' || key === 'pop' ? { util: m?.gpu_util_percent } : e.watchdog?.gpu?.[key];
    const acts = activity(snap())[key] || [];
    const idle = gpu && num(gpu.util) && gpu.util < 5 && !acts.length, down = gpu && gpu.error;
    const tone = down ? 'crit' : idle ? 'warn' : 'ok';
    const el = h('button', { class: `mini-machine ${tone}`, type: 'button', onclick: () => go('machines'), 'aria-label': `${title}: ${down ? 'unreachable' : num(gpu?.util) ? `GPU ${gpu.util}%` : ''}` },
      h('div', { class: 'mm-head' }, h('b', {}, title), h('span', { class: `gpu-dot ${tone}` }, down ? '✖ unreachable' : num(gpu?.util) ? `GPU ${Math.round(gpu.util)}%` : 'GPU —')),
      acts.length ? h('ul', {}, acts.slice(0, 3).map(a => h('li', {}, a.text)), acts.length > 3 ? h('li', { class: 'muted' }, `+${acts.length - 3} more`) : null) : h('div', { class: 'muted small' }, down ? gpu.error : idle ? 'idle' : 'nothing of ours'));
    if (m) el.append(meter(memorySegments(m).used / memorySegments(m).total), h('div', { class: 'small muted' }, `memory ${gb(memorySegments(m).used, 0)} of ${gb(m.mem_total_gb, 0)}`));
    return el;
  }
  function creditsCard(e) {
    const c = e.colab;
    if (!c || c.balance == null) return card('Colab credits', null, h('div', { class: 'muted' }, c?.probe_error ? `Could not read the balance: ${c.probe_error}` : 'Reading the balance…'));
    const tone = c.runway_h < 4 ? 'crit' : c.runway_h < 12 ? 'warn' : 'ok';
    const el = card('Colab credits', more('Machines', 'machines'),
      h('div', { class: 'tiles' }, tile(`${c.balance.toFixed(0)}`, 'units left'), tile(`${c.rate}/h`, 'burning now'), tile(dur(c.runway_h * 3600), 'runway', tone)),
      h('div', { class: 'small', style: 'margin-top:6px' }, `At this rate the balance is gone ≈ ${dayClock(c.empty_ts)}. ${tone === 'ok' ? '' : 'Refill before then. '}`, c.assignments != null ? `${c.assignments} session${c.assignments === 1 ? '' : 's'} assigned.` : ''));
    if ((c.hist || []).length > 3) el.append(h('div', { class: 'small muted', style: 'margin-top:6px' }, 'balance, recent'), spark(c.hist.map(x => x[1]), { label: 'Colab balance', slot: 3 }));
    return el;
  }
  function changesCard(e) {
    const evs = [...(e.events || [])].filter(x => !['driver'].includes(x.kind) || x.level !== 'info' || true).sort((a, b) => b.ts - a.ts).slice(0, 8);
    const el = card('What changed recently', more('All rounds', 'training'));
    el.append(h('div', { class: 'timeline' }, evs.map(ev => h('div', { class: `ev ${ev.level === 'warn' ? 'warn' : ''} ${ev.level === 'ok' ? 'okev' : ''}` }, h('time', { title: dayClock(ev.ts) }, ago(ev.ts, now())), h('span', { class: 't' }, ev.text)))));
    if ((e.decisions || []).length) el.append(h('details', {}, h('summary', {}, 'Latest decisions'), h('ul', { class: 'plain small' }, e.decisions.slice(0, 5).map(d => h('li', {}, h('span', { class: 'muted' }, d.when.slice(5)), ' ', d.title)))));
    return el;
  }
  function overview(hv) {
    const e = ei(), v = h('div', { class: 'now' });
    if (!snap()) return v.append(h('div', { class: 'card muted' }, 'Loading…')), v;
    if (!e) return v.append(h('div', { class: 'notice' }, 'The exporter running on the DGX has no expert-iteration section yet. Restart it from the current checkout.')), v;
    v.append(statusCard(hv, e));
    const al = alertsView(hv);
    if (al) v.append(al);
    const train = latestRound('sft') || latestRound('ready'), prep = (e.rounds || []).filter(r => r.state === 'preparing').at(-1);
    if (train) v.append(roundCard(train, { compactCard: true }));
    if (prep) v.append(roundCard(prep, { compactCard: true }));
    if (!train && !prep) v.append(h('div', { class: 'card muted' }, 'No round in flight.'));
    v.append(progressCard(e), freshDataCard(e));
    v.append(section('Machines', 'tap one for its history'));
    v.append(h('div', { class: 'machine-grid' }, machineMini('dgx', 'DGX', e), machineMini('colab-g4a', 'Colab A100', e), machineMini('pop', 'Pop', e), machineMini('colab-g4b', 'Colab L4', e)));
    v.append(creditsCard(e), changesCard(e));
    return v;
  }

  /* ---------- Rounds ---------- */
  function timelineCard(e) {
    const rows = (e.rounds || []).slice(-3).map(r => ({ label: r.id, segments: roundSegments(r, now()) })).filter(r => r.segments.length);
    if (!rows.length) return null;
    return card('Round timeline', null, h('div', { class: 'small muted', style: 'margin:-4px 0 6px' }, 'Preparing the next round overlaps the running SFT.'), gantt(rows, { now: now(), stages: STAGES }),
      h('div', { class: 'panel-note' }, 'Hatched parts are estimates: the SFT end from the learner\'s step rate, the publish from the typical merge and export time.'));
  }
  const when = ts => !num(ts) ? '—' : new Date(ts).toDateString() === new Date(now()).toDateString() ? clock(ts) : dayClock(ts).replace(/, /, ' ');
  function versionTable(e) {
    const results = e.held?.results || [];
    const rows = (e.lineage || []).map(l => {
      const r = (e.rounds || []).find(x => x.id === l.round), res = results.filter(x => x.version === l.version).at(-1);
      return h('tr', {}, h('td', {}, h('b', {}, `v${l.version}`)), h('td', {}, l.round || e.reference?.name || 'start'), h('td', {}, when(r?.stamps?.published)), h('td', { class: 'n' }, res ? rate(res.rate) : '—'), h('td', { class: 'n' }, r?.health?.pass_rate != null ? rate(r.health.pass_rate) : '—'));
    });
    return card('Policy versions', null, h('div', { class: 'scroll', style: 'max-height:none' }, h('table', { class: 'nowrap' }, h('thead', {}, h('tr', {}, ['version', 'round', 'published', 'held', 'collect'].map((c, i) => h('th', { class: i > 2 ? 'n' : '' }, c)))), h('tbody', {}, rows))),
      h('div', { class: 'panel-note' }, 'held: held-v3 pass rate (only comparable within one measurement series, see Held-out). collect: the policy\'s own pass rate on training tasks while it collected.'));
  }
  function mixBlock(r) {
    const mix = r.mix;
    if (!mix) return null;
    const parts = Object.entries(mix.parts || {}).filter(([, p]) => p.loss_share).sort((a, b) => b[1].loss_share - a[1].loss_share);
    const seg = parts.map(([k, p]) => ({ label: SOURCES[k]?.[0] || k, value: p.loss_share, color: slotVar(SOURCES[k]?.[1] ?? 1) }));
    const newV = Object.entries(mix.parts || {}).filter(([, p]) => p.kept != null && p.kept > 0);
    const kept = newV.reduce((a, [, p]) => a + p.kept, 0), unseen = newV.reduce((a, [, p]) => a + Math.min(p.kept, p.unseen || 0), 0), repeated = newV.reduce((a, [, p]) => a + (p.repeated || 0), 0);
    return h('div', { class: 'mix' }, h('h3', {}, 'What the round trains on'), segBar(seg, { height: 16, label: 'Share of training loss by source' }),
      h('div', { class: 'legend', style: 'margin-top:6px' }, seg.map(s => h('span', { class: 'key' }, h('i', { style: `background:${s.color};height:10px;width:10px` }), `${s.label} ${Math.round(100 * s.value)}%`))),
      kept ? h('div', { class: 'small', style: 'margin-top:8px' }, h('b', {}, `${compact(kept)} episodes kept`), `: ${compact(repeated)} seen in an earlier round, ${compact(kept - repeated)} new to training${unseen < kept - repeated ? '' : ''}.`) : null,
      kept ? segBar([{ label: 'seen in an earlier round', value: repeated, color: 'var(--muted-fill)' }, { label: 'new to training', value: kept - repeated, color: slotVar(2) }], { height: 10, label: 'Repeated versus new episodes' }) : null,
      h('div', { class: 'small muted', style: 'margin-top:6px' }, `${compact(mix.chains)} chains, ${compact(mix.forward_tokens)} forward tokens, ${compact(mix.target_tokens)} loss tokens${mix.binding ? `; capped by ${mix.binding} budget (kept ${Math.round(100 * (mix.keep_fraction || 0))}%)` : ''}.`));
  }
  function sftBlock(r) {
    const s = r.sft;
    if (!s) return null;
    const loss = [{ id: 'loss', label: 'SFT loss', slot: 2, points: s.loss.map(([x, y]) => ({ x, y })) }];
    const val = s.val.length > 1 ? [{ id: 'val', label: 'validation loss', slot: 1, points: s.val.map(([x, y]) => ({ x, y })) }] : null;
    return h('div', { class: 'sft' }, h('h3', {}, 'SFT'),
      h('div', { class: 'kv small' }, h('span', {}, 'steps'), h('span', {}, `${s.step} / ${s.of}`), h('span', {}, 'learner speed'), h('span', {}, `${s.chain_tok_s ? compact(s.chain_tok_s) + ' tokens/s' : '—'}${s.s_per_step ? `, ${s.s_per_step} s/step` : ''}`),
        h('span', {}, 'loss, start → now'), h('span', {}, `${fmt(s.loss_first)} → ${fmt(s.loss_last)}`), val ? [h('span', {}, 'validation, start → now'), h('span', {}, `${fmt(s.val_first, 4)} → ${fmt(s.val_last, 4)}`)] : null),
      s.loss.length > 2 ? chartPanel({ key: `sft:${r.id}:loss`, title: 'SFT loss per step (decision targets)', series: loss, extra: { height: 150, smooth: 0.15 }, note: 'Neuralese consumer steps (loss about 6) are a different objective and are left out.' }) : null,
      val ? chartPanel({ key: `sft:${r.id}:val`, title: 'Validation loss (held-out teacher episodes of this round)', series: val, extra: { height: 130, smooth: 0 }, note: 'Each round has its own validation set: values are comparable within a round only.' }) : null);
  }
  function roundDetail(r, open) {
    const st = r.stamps, size = r.size || {}, hs = r.health;
    const timings = [['prepare (frontier, data, build)', st.start, st.build], ['wait for the GPU', st.build, st.sft_start], ['SFT', st.sft_start, st.sft_end], ['merge and publish', st.sft_end, st.published]].filter(([, a, b]) => a && b && b > a);
    const sum = h('summary', {}, h('span', { class: 'rsum' }, h('b', {}, r.id), ' ', r.version != null ? chip(`v${r.version}`, 'ok') : chip({ sft: 'training', preparing: 'preparing', ready: 'built' }[r.state] || r.state, r.state === 'sft' ? 'ok' : 'open'), ' ', h('span', { class: 'muted small' }, r.state === 'published' ? `published ${ago(st.published, now())}` : '')));
    return h('details', { class: 'card rdetail', open: open || undefined }, sum, stepsStrip(r), roundFacts(r),
      timings.length ? h('div', { class: 'kv small', style: 'margin-top:6px' }, timings.map(([l, a, b]) => [h('span', {}, l), h('span', {}, dur((b - a) / 1000))])) : null, mixBlock(r), sftBlock(r),
      hs ? h('div', { class: 'health' }, h('h3', {}, 'Collection health when the round was built'), h('div', { class: 'small' }, `${compact(hs.episodes)} own episodes over ${hs.tasks} tasks, pass rate ${rate(hs.pass_rate)}${hs.prev_pass_rate != null ? ` (previous policy ${rate(hs.prev_pass_rate)})` : ''}.`),
        hs.flags?.length ? h('div', { class: 'row', style: 'margin-top:4px' }, hs.flags.map(f => chip(f.replace(/:$/, ''), 'warn'))) : null) : null,
      r.frontier?.tiers ? h('div', { class: 'frontier' }, h('h3', {}, 'Frontier the round targets'), segBar(TIERS.map(([k, l, s]) => ({ label: l, value: r.frontier.tiers[k] || 0, color: slotVar(s) })), { height: 12, label: 'Frontier tiers' }),
        h('div', { class: 'legend', style: 'margin-top:4px' }, TIERS.filter(([k]) => r.frontier.tiers[k]).map(([k, l, s]) => h('span', { class: 'key' }, h('i', { style: `background:${slotVar(s)};height:10px;width:10px` }), `${l} ${r.frontier.tiers[k]}`)))) : null);
  }
  function training() {
    const e = ei(), v = h('div', {});
    if (!snap()) return v.append(loading('')), v;
    if (!e) return v.append(h('div', { class: 'notice' }, 'The exporter has no expert-iteration section yet.')), v;
    const t = timelineCard(e);
    if (t) v.append(t);
    v.append(versionTable(e));
    v.append(section('Rounds', 'newest first'));
    [...e.rounds].reverse().forEach((r, i) => v.append(roundDetail(r, i < 2)));
    v.append(h('div', { class: 'row', style: 'margin-top:10px' }, h('a', { class: 'btn', href: dl.ds ? '#data' : '#data' }, 'Browse the training data')));
    return v;
  }

  /* ---------- Data flow ---------- */
  function teacherCard(te, colors) {
    const parts = outcomeParts(te).map(p => ({ label: p.label, value: p.n, color: colors[p.key] }));
    return h('div', { class: 'card teacher' }, h('header', {}, h('h2', { class: 'mono' }, te.id.replace(/^[^/]+\//, '')), te.success_rate != null ? chip(`${Math.round(100 * te.success_rate)}% success`, 'ok') : chip('no episodes yet')),
      h('div', { class: 'row small' }, te.provider ? chip(te.provider) : null, te.reasoning ? chip(`reasoning: ${te.reasoning}`, '', 'What the provider returns of the model\'s reasoning: summary, hidden or raw') : null, te.effort ? chip(`effort ${te.effort}`) : null),
      !te.episodes && te.deferred?.unresolved ? h('div', { class: 'small' }, chip(`▲ ${te.deferred.unresolved} attempts unresolved: no reply from the provider`, 'warn')) : null,
      segBar(parts, { height: 14, label: `${te.id} outcomes` }), h('div', { class: 'legend', style: 'margin-top:6px' }, parts.filter(p => p.value).map(p => h('span', { class: 'key' }, h('i', { style: `background:${p.color};height:10px;width:10px` }), `${p.label} ${p.value}`))),
      h('div', { class: 'kv small', style: 'margin-top:6px' }, h('span', {}, 'tasks / episodes'), h('span', {}, `${te.tasks} / ${te.episodes}`), h('span', {}, 'seconds per task'), h('span', {}, fmt(te.seconds_mean, 3)), h('span', {}, 'deferred (retry)'), h('span', {}, Object.entries(te.deferred || {}).map(([k, n]) => `${n} ${k.replace(/_/g, ' ')}`).join(', ') || 'none'), te.provider_failures ? [h('span', {}, 'provider failures'), h('span', {}, te.provider_failures)] : null));
  }
  const OUTCOME_COLORS = { task_success: 'var(--s3)', semantic_mismatch: 'var(--s4)', budget_exhaustion: 'var(--s8)', other: 'var(--muted-fill)' };
  function flow() {
    const e = ei(), v = h('div', {});
    if (!snap()) return v.append(loading('')), v;
    const c = e?.collection, t = snap().teachers;
    v.append(section('Own episodes', 'collected by the newest policy'));
    if (!c) v.append(h('div', { class: 'card muted' }, 'No collection yet.'));
    else {
      v.append(card('Episodes per hour, by machine', null, barStack(hourlyBars(c.hourly, now(), 24), { keys: HOSTS, height: 160 }), h('div', { class: 'panel-note' }, 'Counted when a group of 4 samples finishes. Collection pauses while its machine runs an SFT or a held evaluation.')));
      v.append(h('div', { class: 'grid g2' }, (c.actors || []).map(a => h('div', { class: 'card' }, h('header', {}, h('h2', {}, `${where(a.host === 'colab' ? 'colab' : a.host)} · v${a.version}`), chip('collecting', 'ok')),
        h('div', { class: 'tiles' }, tile(`${a.rate_h}`, 'episodes, last hour'), tile(a.pass_rate_h != null ? rate(a.pass_rate_h) : '—', 'pass, last hour'), tile(a.in_flight ?? '—', 'running now')),
        h('div', { class: 'kv small', style: 'margin-top:8px' }, h('span', {}, 'groups this policy'), h('span', {}, `${a.groups} (${compact(a.episodes)} episodes)`), h('span', {}, 'tasks left in the pool'), h('span', {}, compact(a.tasks_left)),
          a.actor ? [h('span', {}, 'actor decode'), h('span', {}, `${compact(a.actor.tok_s)} tokens/s, ${a.actor.running} requests`)] : null, h('span', {}, 'last group'), h('span', {}, ago(a.last_group_ts, now())))))));
      if (!(c.actors || []).length) v.append(h('div', { class: 'card muted' }, 'No collector is running right now.'));
      if (c.fill?.info) v.append(card('Student fill', null, h('div', { class: 'small' }, `v${c.fill.version}: ${compact(c.fill.info.done)} of ${compact(c.fill.info.episodes)} teacher episodes re-rendered with the student's own reasoning (${compact(c.fill.info.todo)} left).`), meter(c.fill.info.done / c.fill.info.episodes, 'good')));
      const split = p => ['dgx', 'colab', 'pop'].filter(k => p.hosts[k]?.episodes).map(k => `${HOSTS.find(x => x.key === k).label} ${compact(p.hosts[k].episodes)}`).join(' · ');
      v.append(card('By policy', null, h('div', { class: 'scroll', style: 'max-height:none' }, h('table', {}, h('thead', {}, h('tr', {}, ['policy', 'episodes', 'pass', 'where'].map((x, i) => h('th', { class: i === 1 || i === 2 ? 'n' : '' }, x)))),
        h('tbody', {}, [...c.policies].reverse().map(p => h('tr', {}, h('td', {}, h('b', {}, `v${p.version}`)), h('td', { class: 'n' }, compact(p.episodes)), h('td', { class: 'n' }, rate(p.pass_rate)), h('td', { class: 'small' }, split(p))))))),
        h('div', { class: 'panel-note' }, 'Pass rate is successes over scored episodes (episodes lost to infrastructure errors are not counted).')));
    }
    v.append(section('Teachers', t ? `campaign for ${t.pool.replace('teacher-', 'round ')}` : ''));
    if (!t) v.append(h('div', { class: 'card muted' }, 'No teacher campaign yet.'));
    else {
      v.append(h('div', { class: 'grid g4' }, [[`${t.tasks_done}/${t.eligible ?? '?'}`, 'frontier tasks attempted'], [fmt(t.success), 'verified successes'], [`${t.tasks_won ?? '?'}/${t.tasks_done}`, 'tasks with a verified success'], [`${t.deferred}`, 'deferred for retry']].map(([a, b]) => h('div', { class: 'card' }, stat(a, b)))));
      const live = (t.teachers || []).filter(x => x.episodes || Object.keys(x.deferred || {}).length);
      v.append(h('div', { class: 'grid g2' }, live.map(te => teacherCard(te, OUTCOME_COLORS))));
      const idle = (t.teachers || []).filter(x => !live.includes(x));
      if (idle.length) v.append(h('div', { class: 'small muted' }, `Configured but not used yet: ${idle.map(x => x.id).join(', ')}.`));
      if ((t.campaigns || []).length > 1) v.append(card('Campaigns by round', null, h('div', { class: 'scroll', style: 'max-height:none' }, h('table', {}, h('thead', {}, h('tr', {}, ['round', 'tasks', 'successes', 'tasks won'].map((x, i) => h('th', { class: i ? 'n' : '' }, x)))),
        h('tbody', {}, [...t.campaigns].reverse().map(c => h('tr', {}, h('td', {}, c.pool.replace('teacher-', '')), h('td', { class: 'n' }, c.tasks_done), h('td', { class: 'n' }, c.success), h('td', { class: 'n' }, c.tasks_done && c.tasks_won != null ? `${Math.round(100 * c.tasks_won / c.tasks_done)}%` : '—'))))))));
      v.append(h('div', { class: 'row' }, more('Teacher details and families', 'teachers')));
    }
    const reg = e?.registry || [];
    if (reg.length) v.append(section('Newly registered datasets', 'data registry'), card('Latest registry entries', h('a', { class: 'btn sm', href: '#data' }, 'Browse the training data'), h('ul', { class: 'plain small' }, reg.map(r => h('li', {}, h('span', { class: 'muted' }, ago(r.ts, now())), ' ', h('span', { class: 'mono' }, r.id), ' ', chip(r.kind || '?'), r.rows != null ? h('span', { class: 'muted' }, ` ${compact(Number(r.rows))} rows`) : null)))));
    return v;
  }

  /* ---------- Held-out ---------- */
  function resultRow(r) {
    const ci = r.task_se != null ? ` ±${(196 * r.task_se).toFixed(1)}` : '';
    return h('div', { class: 'mrow' }, h('div', { class: 'row between' }, h('b', {}, r.label), h('span', { class: 'big2' }, `${(100 * r.rate).toFixed(1)}%`, h('span', { class: 'muted small' }, ci))),
      h('div', { class: 'row small muted' }, `${r.successes}/${r.scored} scored of ${r.ran}`, ' · ', chip(`series ${String(r.series).slice(0, 4)}`), r.infra ? chip(`${r.infra} lost to infrastructure`, 'warn') : null, r.ran_out_rate != null ? chip(`ran out ${rate(r.ran_out_rate)}`) : null, r.neuralese_use != null ? chip(`Neuralese ${rate(r.neuralese_use)}`) : null, r.unpaired ? chip('unpaired copy') : null, ' ', ago(r.finished, now())));
  }
  function held() {
    const e = ei(), v = h('div', {});
    if (!snap()) return v.append(loading('')), v;
    const hd = e?.held;
    v.append(h('div', { class: 'notice' }, 'Held-v3 is 256 tasks, 2 samples each, sampled at temperature 1. Only results of one measurement series (same scoring code, engine and sampling) are paired; the table says which series a result belongs to. Interval: 1.96 standard errors over tasks.'));
    if (hd) {
      const g = hd.gate, prog = progress(e);
      v.append(card('The gate', null, h('div', { class: `verdict tone-${prog.tone}` }, h('b', {}, prog.headline), h('span', {}, ` ${prog.sub}`)), h('div', { class: 'small' }, prog.detail),
        g ? h('div', { class: 'kv small', style: 'margin-top:8px' }, h('span', {}, 'candidate'), h('span', {}, heldLine(g)), h('span', {}, 'reference'), h('span', {}, g.reference_same_series ? `${g.reference} measured on the same series` : `${g.reference}: not yet measured on the candidate's series`),
          h('span', {}, 'cadence'), h('span', {}, `every ${g.every ?? '?'} round${g.every === 1 ? '' : 's'} (${g.since_held ?? '?'} since the last)`), g.failing ? [h('span', {}, 'retries'), h('span', {}, `${g.failing.count}× since ${clock(g.failing.first_ts * 1000)}`)] : null) : null));
      for (const p of hd.in_progress || []) v.append(card(`Measuring ${p.label} on Pop`, h('span', { class: 'muted small' }, `updated ${ago(p.updated, now())}`), meter(p.done / p.of, 'good'), h('div', { class: 'small' }, `${p.done} of ${p.of} episodes · ${dur((p.minutes || 0) * 60)} so far · ${p.eta_s ? `about ${dur(p.eta_s)} left (${clock(now() + p.eta_s * 1000)})` : ''}`)));
      const pts = hd.results.map(r => ({ label: r.label.replace(/^v(\d+) \(.*\)$/, 'v$1 ref'), y: r.rate, lo: r.task_se != null ? r.rate - 1.96 * r.task_se : null, hi: r.task_se != null ? r.rate + 1.96 * r.task_se : null, series: r.series }));
      v.append(card('Pass rate by policy', null, pts.length ? ladder(pts, { height: 220 }) : h('div', { class: 'muted' }, 'No result yet.')));
      if ((hd.pairs || []).length) v.append(card('Paired comparisons (same series)', null, h('div', { class: 'scroll', style: 'max-height:none' }, h('table', {}, h('thead', {}, h('tr', {}, ['pair', 'change', '95% interval'].map((c, i) => h('th', { class: i ? 'n' : '' }, c)))),
        h('tbody', {}, [...hd.pairs].reverse().map(p => h('tr', { class: p.ci_pts[0] > 0 ? 'sig-good' : p.ci_pts[1] < 0 ? 'sig-bad' : '' }, h('td', {}, p.a, h('span', { class: 'muted' }, ' → '), p.b, h('div', { class: 'small muted' }, `${p.up} tasks up, ${p.down} down`)), h('td', { class: 'n' }, h('b', {}, signed(p.delta_pts, 1)), ' pts'), h('td', { class: 'n nowrap' }, `[${signed(p.ci_pts[0], 1)}, ${signed(p.ci_pts[1], 1)}]`)))))),
        h('div', { class: 'panel-note' }, 'Task-level normal approximation of the per-task pass-rate change. The driver\'s verdict uses the permutation test of held_eval.')));
      v.append(section('Every measurement'), h('div', { class: 'card mlist' }, [...hd.results].reverse().map(resultRow)));
      if ((hd.queue || []).length) v.append(card('held-v4 queue', null, h('ul', { class: 'plain small' }, hd.queue.map(q => h('li', {}, `${q.label} · ${q.state} · ${(q.reasons || []).join(', ')} · queued ${ago(q.queued_at, now())}`)))));
    }
    const legacy = (snap().runs || []).filter(r => (r.held || []).length);
    if (legacy.length) v.append(h('details', { class: 'card', style: 'margin-top:14px' }, h('summary', {}, `Earlier GRPO runs (retired): ${legacy.length} with held results`), h('table', {}, h('thead', {}, h('tr', {}, ['run', 'step', 'change', 'p'].map(c => h('th', {}, c)))),
      h('tbody', {}, legacy.flatMap(r => r.held.map(x => h('tr', {}, h('td', {}, h('a', { href: dl.ds('grpo:' + r.id.replace(/\./g, '/'), { part: 'held' }) }, r.id)), h('td', {}, x.step), h('td', {}, `${signed(x.delta_pts, 1)} pts`), h('td', {}, fmt(x.p, 2)))))))));
    return v;
  }

  /* ---------- Machines ---------- */
  const histSeries = (e, name, idx, label, slot) => ({ id: label, label, slot, points: (e.hist?.[name] || []).filter(r => num(r[idx])).map(r => ({ x: r[0], y: r[idx] })) });
  function machineDetail(key, title, subtitle, e) {
    const m = snap().machines?.[key], acts = activity(snap())[key] || [];
    const gpu = e?.watchdog?.gpu?.[key === 'dgx' || key === 'pop' ? key : key];
    const el = h('div', { class: 'card' }, h('header', {}, h('h2', {}, title), h('span', { class: 'muted small' }, subtitle)));
    el.append(acts.length ? h('ul', { class: 'acts' }, acts.map(a => h('li', {}, a.text))) : h('div', { class: 'muted small' }, gpu?.error ? gpu.error : 'Nothing of ours is running here.'));
    if (m) {
      const ms = memorySegments(m), parts = [...ms.segs.filter(s => s.kind === 'claim').map((s, i) => ({ label: s.label, value: s.gb, color: slotVar(i) })), ...ms.segs.filter(s => s.kind === 'other').map(s => ({ label: s.label, value: s.gb, color: 'var(--muted-fill)' }))];
      el.append(h('div', { class: 'row between', style: 'margin-top:8px' }, h('span', {}, h('b', {}, gb(ms.used)), h('span', { class: 'muted' }, ` used of ${gb(ms.total, 0)}`)), h('span', { class: 'muted small' }, `${gb(m.mem_available_gb)} available`)), segBar(parts, { total: ms.total, height: 14, label: 'Memory by ledger claim' }));
      el.append(h('details', { class: 'small' }, h('summary', {}, `${parts.length} memory claims`), h('div', { class: 'legend' }, parts.map(p => h('span', { class: 'key' }, h('i', { style: `background:${p.color};height:10px;width:10px` }), `${p.label} ${gb(p.value)}`)))));
      el.append(h('div', { class: 'tiles', style: 'margin-top:10px' }, tile(pct(m.cpu_percent), 'CPU'), tile(pct(m.gpu_util_percent), 'GPU util'), key === 'dgx' ? tile(gb(m.ledger?.headroom_gb), `ledger headroom (floor 15)`, num(m.ledger?.headroom_gb) && m.ledger.headroom_gb < 15 ? 'crit' : '') : tile(gb(m.ledger?.headroom_gb), 'ledger headroom')));
      for (const d of m.disks || []) {
        const floor = d.floor_gb, low = num(floor) && d.free_gb < floor, near = num(floor) && d.free_gb < floor + 20, fall = e?.disks?.[`${key}:${d.mount}`]?.rate_gb_h;
        el.append(h('div', { class: 'disk' }, h('div', { class: 'row between' }, h('span', {}, d.mount), h('span', { class: low ? 'tone-crit' : near ? 'tone-warn' : 'muted' }, `${d.free_gb} GB free of ${d.size_gb}${num(floor) ? ` · floor ${floor}` : ''}${low ? ' ✖' : near ? ' ▲' : ''}`)), meter((d.size_gb - d.free_gb) / d.size_gb, low ? 'crit' : near ? 'warn' : ''),
          num(fall) && Math.abs(fall) >= 0.5 ? h('div', { class: 'small muted' }, `${fall > 0 ? 'falling' : 'rising'} ${Math.abs(fall)} GB/h over the last 3 h${fall > 0 && num(floor) ? `; ${Math.max(0, (d.free_gb - floor) / fall).toFixed(1)} h to the floor` : ''}`) : null));
      }
    } else if (gpu) el.append(h('div', { class: 'tiles', style: 'margin-top:10px' }, tile(num(gpu.util) ? `${Math.round(gpu.util)}%` : '—', 'GPU util'), tile(gpu.idle_s != null ? dur(gpu.idle_s) : '—', 'idle for', gpu.idle_s > 1800 ? 'warn' : '')));
    const g = histSeries(e || {}, key, 2, 'GPU %', 2), mem = histSeries(e || {}, key, 1, 'Memory GB', 0);
    if (g.points.length > 2) el.append(h('div', { class: 'grid', style: 'grid-template-columns:1fr 1fr;gap:12px;margin-top:10px' }, mem.points.length > 2 ? h('div', {}, h('div', { class: 'small muted' }, 'memory used, 24 h'), spark(mem.points.map(p => p.y), { label: 'memory', slot: 0 })) : null, h('div', {}, h('div', { class: 'small muted' }, 'GPU util, 24 h'), spark(g.points.map(p => p.y), { label: 'GPU', slot: 2 }))));
    return el;
  }
  function machines() {
    const e = ei(), v = h('div', {});
    if (!snap()) return v.append(loading('')), v;
    v.append(h('div', { class: 'notice' }, 'The GB10 shares one memory pool between CPU and GPU: "GPU memory" is system RAM, so the bars show ledger claims against total RAM. Ledger headroom is available memory minus what running claims still reserve.'));
    v.append(h('div', { class: 'grid g2' }, machineDetail('dgx', 'DGX', 'GB10, unified memory', e), machineDetail('pop', 'Pop', 'RTX 4060 8 GB', e), machineDetail('colab-g4a', 'Colab A100', 'round learner and collection', e), machineDetail('colab-g4b', 'Colab L4', 'program tuning', e)));
    v.append(creditsCard(e || {}));
    const c = e?.colab;
    if (c) {
      if ((c.hist || []).length > 3) v.append(chartPanel({ key: 'credits', title: 'Credit balance (units)', series: [{ id: 'bal', label: 'balance', slot: 3, points: c.hist.map(x => ({ x: x[0], y: x[1] })) }], xKind: 'time', xFmt: clock, xLabel: '', extra: { height: 160, smooth: 0, gap: 3 * 3600e3, yMin: 0 }, note: 'A rise is a refill.' }));
      if ((c.spent_sessions || []).length) {
        const rows = [...c.spent_sessions].reverse().map(s => h('tr', {}, h('td', {}, dayClock(s.started)), h('td', { class: 'n' }, s.hours), h('td', { class: 'n' }, fmt(s.cu, 3)), h('td', { class: 'n' }, s.cu_per_hour ?? '—')));
        const head = h('tr', {}, ['started', 'hours', 'units', 'per hour'].map((x, i) => h('th', { class: i ? 'n' : '' }, x)));
        v.append(card('Credits spent by session', null, h('div', { class: 'scroll', style: 'max-height:none' }, h('table', {}, h('thead', {}, head), h('tbody', {}, rows))), h('div', { class: 'panel-note' }, `${fmt(c.spent_cu, 3)} units spent by this driver so far.`)));
      }
    }
    const wd = e?.watchdog?.alerts || [];
    v.append(section('Watchdog', e?.watchdog?.checked ? `checked ${ago(e.watchdog.checked, now())}` : ''), wd.length ? h('div', { class: 'card' }, wd.map(a => h('div', { class: 'wd' }, chip(a.severity === 'critical' ? '✖ critical' : '▲ warning', a.severity === 'critical' ? 'crit' : 'warn'), ' ', a.message, h('span', { class: 'muted small' }, ` · ${ago(a.last_ts, now())}`)))) : h('div', { class: 'card muted' }, 'No active watchdog alerts.'));
    const tu = e?.tuning;
    if (tu) v.append(card('Program tuning', null, tu.running.length ? h('ul', { class: 'plain small' }, tu.running.map(u => h('li', {}, `${u.title} · ${u.machine} · ${dur(u.elapsed_s)}`))) : h('div', { class: 'muted small' }, 'No tuning unit is running.'), tu.last_log ? h('div', { class: 'log mono', title: tu.last_log }, tu.last_log) : null));
    return v;
  }

  /* ---------- Units ---------- */
  const ROLE_ORDER = ['driver', 'sft', 'collect', 'fill', 'prepare', 'teacher', 'held', 'tuning', 'taskqual', 'datafill', 'colab', 'housekeeping', 'plumbing', 'test', 'other'];
  const ROLE_NAMES = { driver: 'Round driver', sft: 'SFT', collect: 'Collection', fill: 'Student fill', prepare: 'Round preparation', teacher: 'Teacher campaign', held: 'Held evaluation', tuning: 'Program tuning', taskqual: 'Task qualification', datafill: 'Vibestudio student data', colab: 'Colab sessions', housekeeping: 'Housekeeping', plumbing: 'Ops plumbing', test: 'Tests', other: 'Other' };
  function unitRow(p) {
    const run = p.state === 'running', res = p.resources || {};
    return h('details', { class: `urow ${p.state}` }, h('summary', {}, h('span', { class: `dotstate ${run ? 'ok' : p.state === 'failed' ? 'crit' : ''}`, 'aria-hidden': 'true' }), h('b', {}, p.title), ' ', machineChip(p.machine), h('span', { class: 'mono muted ellip-unit' }, shortUnit(p.unit).replace(/^aei1-/, '')), h('span', { class: 'muted small' }, ` ${run ? dur(p.elapsed_s) : `${p.state} ${ago(p.started_ts + (p.elapsed_s || 0) * 1000, now())}`}`)),
      h('div', { class: 'small' }, p.description), p.progress ? [meter(p.progress.total ? p.progress.done / p.progress.total : 0, 'good'), h('div', { class: 'small muted' }, `${fmt(p.progress.done)}${p.progress.total ? ` / ${fmt(p.progress.total)}` : ''} ${p.progress.unit} · ${etaText(p.eta_s ? now() + p.eta_s * 1000 : null, now())}`)] : null,
      num(res.budget_gb) ? h('div', { class: 'small muted' }, `memory ${gb(res.used_gb)} used, ${gb(res.peak_gb)} peak, ${gb(res.budget_gb, res.budget_gb < 10 ? 1 : 0)} budget${res.gpu ? ', GPU' : ''}${num(res.cpu_percent) ? `, ${res.cpu_percent}% CPU` : ''}`) : null,
      p.last_log ? h('div', { class: 'log mono', title: p.last_log }, p.last_log) : null, h('div', { class: 'mono small muted' }, shortUnit(p.unit)),
      h('div', { class: 'row' }, askBtn('unit', p.unit, p.title), run ? h('button', { class: 'btn sm danger', type: 'button', onclick: () => ctx.openAsk({ mode: 'stop', subject: { kind: 'unit', id: p.unit, title: p.title } }) }, 'Request stop') : null, p.run ? runLink(p.run) : null));
  }
  function units() {
    const v = h('div', {}), all = snap()?.processes || [];
    if (!snap()) return v.append(loading('')), v;
    const seg = (key, opts) => h('span', { class: 'seg-ctl' }, opts.map(([val, label]) => h('button', { type: 'button', 'aria-pressed': S[key] === val, onclick: () => { S[key] = val; ctx.render(); } }, label)));
    v.append(h('div', { class: 'toolbar' }, seg('procMachine', [['all', 'Both'], ['dgx', 'DGX'], ['pop', 'Pop']]), seg('procState', [['live', 'Running'], ['done', 'Failed / finished'], ['all', 'All']])));
    const keep = p => (S.procMachine === 'all' || p.machine === S.procMachine) && (S.procState === 'all' || (S.procState === 'live') === (p.state === 'running'));
    const rows = all.filter(keep);
    for (const role of ROLE_ORDER) {
      const mine = rows.filter(p => (p.role || 'other') === role);
      if (!mine.length) continue;
      const failed = mine.filter(p => p.state === 'failed').length;
      v.append(h('details', { class: 'card ugroup', open: S.procState === 'live' || role === 'driver' || undefined }, h('summary', {}, h('b', {}, ROLE_NAMES[role]), ' ', chip(`${mine.length}`), failed ? chip(`${failed} failed`, 'crit') : null), h('div', { class: 'ulist' }, mine.map(unitRow))));
    }
    if (!rows.length) v.append(h('div', { class: 'card muted' }, 'No matching units.'));
    v.append(section('Memory ledger claims', 'budget versus use'));
    for (const [name, m] of Object.entries(snap().machines || {})) {
      const claims = m.ledger?.claim_list || [];
      v.append(h('details', { class: 'card', style: 'margin-bottom:10px' }, h('summary', {}, h('b', {}, name === 'dgx' ? 'DGX ledger' : 'Pop ledger'), h('span', { class: 'muted small' }, ` ${gb(m.ledger?.available_gb)} available · headroom ${gb(m.ledger?.headroom_gb)} · ${m.ledger?.waiting || 0} waiting`)),
        claims.length ? h('div', { class: 'scroll', style: 'max-height:none' }, h('table', {}, h('thead', {}, h('tr', {}, ['unit', 'used', 'peak', 'budget'].map((c, i) => h('th', { class: i ? 'n' : '' }, c)))), h('tbody', {}, claims.map(c => h('tr', {}, h('td', { class: 'mono' }, shortUnit(c.unit)), h('td', { class: 'n' }, gb(c.used_gb)), h('td', { class: 'n' }, gb(c.peak_gb)), h('td', { class: 'n' }, gb(c.budget_gb, c.budget_gb < 10 ? 1 : 0))))))) : h('div', { class: 'muted' }, 'No claims.')));
    }
    return v;
  }

  /* ---------- Teachers (detail) ---------- */
  function teachers() {
    const tt = snap()?.teachers, v = h('div', {});
    if (!tt) return v.append(loading('No teacher campaign yet.')), v;
    v.append(h('div', { class: 'grid g4' }, [[`${tt.tasks_done}/${tt.eligible ?? '?'}`, 'frontier tasks attempted'], [fmt(tt.success), 'verified successes'], [fmt(tt.deferred), 'deferred (retry) rows'], [`${tt.families_with_success}/${tt.families_total}`, 'families with a success'], [`${fmt(tt.success_last_hour)}/h`, 'successes, last hour']].map(([a, b]) => h('div', { class: 'card' }, stat(a, b)))));
    v.append(section('Per teacher', tt.pool), h('div', { class: 'grid g2' }, tt.teachers.map(te => teacherCard(te, OUTCOME_COLORS))));
    const hist = tt.history || [];
    if (hist.length > 2) v.append(section('Progress over time'), chartPanel({ key: 'teach', title: 'Tasks done and verified successes', series: [{ id: 'done', label: 'tasks done', slot: 0, points: hist.map(r => ({ x: r[0], y: r[1] })) }, { id: 'success', label: 'successes', slot: 2, points: hist.map(r => ({ x: r[0], y: r[2] })) }], xKind: 'time', xFmt: clock, xLabel: '', yFmt: x => fmt(x, 4), extra: { smooth: 0, height: 200, gap: 3 * 3600e3 } }));
    v.append(section('Families', `${tt.families_total} seen`), h('div', { class: 'card' }, h('div', { class: 'scroll' }, h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, 'family'), h('th', { class: 'n' }, 'tasks'), h('th', { class: 'n' }, 'successes'))), h('tbody', {}, tt.families.map(f => h('tr', {}, h('td', { class: 'mono' }, f.family), h('td', { class: 'n' }, f.tasks), h('td', { class: 'n' }, f.success))))))));
    return v;
  }

  /* ---------- Open items and fallow ---------- */
  function backlog() {
    const v = h('div', {}), e = ei(), sn = snap() || {};
    v.append(section('In flight', 'plans/semantic-fp/OPEN_ITEMS.md'));
    v.append((e?.open_items || []).length ? h('div', { class: 'card' }, h('ul', { class: 'plain' }, e.open_items.map(i => h('li', { class: 'oi' }, h('div', {}, i.item), h('div', { class: 'small muted' }, i.owner, ' ', askBtn('open-item', i.item.slice(0, 80), i.item.slice(0, 120))))))) : h('div', { class: 'card muted' }, 'Nothing listed.'));
    if ((sn.queue || []).length) {
      v.append(section('Queue', 'what is waiting and what it waits for'));
      v.append(h('div', { class: 'grid g2' }, sn.queue.map(q => h('div', { class: 'card proc' }, h('div', { class: 'phead row' }, h('h3', {}, q.title), q.machine ? machineChip(q.machine) : null), h('div', {}, q.description), h('div', { class: 'row' }, chip(`▲ blocked on: ${q.blocked_on || '—'}`, 'warn'), askBtn('queue', q.title, q.title))))));
    }
    v.append(section('Fallow ledger', 'built capabilities not in use'));
    const rows = (sn.fallow || []).filter(r => matches(r, S.q));
    v.append(h('div', { class: 'toolbar' }, h('input', { type: 'search', placeholder: 'Filter the fallow ledger…', value: S.q, 'aria-label': 'Filter fallow rows', oninput: ev => { S.q = ev.target.value; S.hold = false; clearTimeout(S.qt); S.qt = setTimeout(ctx.render, 250); } }), h('span', { class: 'muted small' }, `${rows.length} of ${(sn.fallow || []).length}`)));
    let sec = null; const body = h('tbody', {});
    for (const r of rows) {
      if (r.section !== sec) { sec = r.section; body.append(h('tr', {}, h('th', { colspan: 4, style: 'background:var(--sunken)' }, sec))); }
      body.append(h('tr', {}, h('td', {}, h('b', {}, r.piece), h('div', { class: 'mono muted' }, r.id)), h('td', {}, r.code_now), h('td', {}, r.blocker), h('td', {}, r.next_action, ' ', askBtn('fallow', r.id, r.piece))));
    }
    v.append(h('div', { class: 'card' }, h('div', { class: 'scroll', style: 'max-height:none' }, h('table', {}, h('thead', {}, h('tr', {}, ['capability', 'code now', 'blocker', 'next action'].map(c => h('th', {}, c)))), body))));
    return v;
  }

  /* ---------- More ---------- */
  function moreView(hv) {
    const items = [['units', 'Units', 'Everything systemd runs, grouped by what it does'], ['teachers', 'Teacher details', 'Per teacher outcomes, families, history'], ['data', 'Training data', 'Browse datasets, tasks, episodes and rounds'], ['backlog', 'Open items and fallow', 'In-flight work and built capabilities not in use'],
      ['investigations', 'Investigations', 'Open questions with findings'], ['inbox', 'Inbox', 'Questions to the supervising agent and its answers'], ['events', 'All events', 'The driver\'s event log']];
    return h('div', { class: 'moremenu' }, items.map(([id, t, d]) => h('a', { class: 'card menuitem', href: `#${id}` }, h('b', {}, t), h('span', { class: 'muted small' }, d))));
  }
  function events() {
    const e = ei(), v = h('div', {});
    const evs = [...(e?.events || [])].sort((a, b) => b.ts - a.ts);
    v.append(h('div', { class: 'card' }, evs.length ? h('div', { class: 'timeline' }, evs.map(ev => h('div', { class: `ev ${ev.level === 'warn' ? 'warn' : ''}` }, h('time', { title: dayClock(ev.ts) }, ago(ev.ts, now())), h('span', { class: 't' }, ev.text)))) : h('div', { class: 'muted' }, snap() ? 'No events yet.' : 'Loading…')));
    return v;
  }

  return { overview, training, flow, held, machines, units, teachers, backlog, more: moreView, events };
}
