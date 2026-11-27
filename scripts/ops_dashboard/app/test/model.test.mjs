// Pure helpers of the ops page (ui/model.mjs): round steps and estimates, the progress verdict, health, activity. Run: node --test test/
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { compact, etaText, roundSteps, roundEta, roundSegments, hourlyBars, progress, activity, health, rate, clockSkew } from '../pages/ops/ui/model.mjs';
import { SIGNAL_CLOCK_SKEW_MS } from '../pages/ops/feeds.mjs';

const NOW = Date.parse('2026-11-24T09:00:00Z');
const round = (over = {}) => ({ id: 'r0004', state: 'sft', version: null, stamps: { start: NOW - 5 * 3600e3, frontier: NOW - 5 * 3600e3, data: NOW - 3 * 3600e3, build: NOW - 2 * 3600e3, sft_start: NOW - 3600e3, sft_end: null, published: null },
  sft: { step: 100, of: 200, eta_ts: NOW + 1800e3 }, sft_running: true, mix: { chains: 40000 }, frontier: { tasks: 46 }, ...over });

test('compact numbers and eta text', () => {
  assert.deepEqual([compact(950), compact(1234), compact(45_000), compact(2_400_000), compact(null)], ['950', '1.2k', '45k', '2.4M', '—']);
  assert.equal(rate(0.6372), '63.7%');
  assert.match(etaText(NOW + 41 * 60e3, NOW), /in 41m/);
  assert.equal(etaText(NOW - 5, NOW), 'finishing');
  assert.equal(etaText(null, NOW), 'ETA unknown');
});

test('round steps follow the stage stamps', () => {
  const s = roundSteps(round(), { results: [] }, NOW);
  assert.deepEqual(s.map(x => x.state), ['done', 'done', 'done', 'done'].slice(0, 3).concat(['active', 'todo', 'todo']));
  const done = roundSteps(round({ state: 'published', version: 4, stamps: { ...round().stamps, sft_end: NOW - 600e3, published: NOW - 300e3 } }), { results: [{ version: 4, rate: 0.6 }] }, NOW);
  assert.deepEqual(done.map(x => x.state), ['done', 'done', 'done', 'done', 'done', 'done']);
  assert.equal(done[5].detail, '60.0%');
  const waiting = roundSteps(round({ state: 'published', version: 4, stamps: { ...round().stamps, sft_end: NOW - 600e3, published: NOW - 300e3 } }), { results: [] }, NOW);
  assert.equal(waiting[5].state, 'active');
  const prep = roundSteps({ id: 'r5', state: 'preparing', stamps: { frontier: NOW - 1000 }, data_progress: 12 }, null, NOW);
  assert.deepEqual(prep.map(x => x.state), ['done', 'active', 'todo', 'todo', 'todo', 'todo']);
});

test('a running round gets an estimated SFT end, publish five minutes later, and hatched timeline parts', () => {
  const r = round(), e = roundEta(r);
  assert.equal(e.sftEnd, NOW + 1800e3);
  assert.equal(e.publish, NOW + 1800e3 + 300e3);
  const segs = roundSegments(r, NOW);
  assert.deepEqual(segs.map(s => [s.key, s.est]), [['prepare', false], ['sft', false], ['sft', true], ['publish', true]]);
  assert.ok(segs.every(s => s.to > s.from));
});

test('hourly bars fill missing hours with zeros and keep the newest last', () => {
  const top = Math.floor(NOW / 3600e3) * 3600e3;
  const bars = hourlyBars([{ t: top, dgx: 5, colab: 0, pop: 1 }, { t: top - 2 * 3600e3, dgx: 7 }], NOW, 4);
  assert.deepEqual(bars.map(b => b.dgx), [0, 7, 0, 5]);
  assert.equal(bars.at(-1).pop, 1);
});

test('progress prefers the paired held result and calls an interval across zero noise', () => {
  const held = { results: [{ version: 1, rate: 0.637, scored: 460, ran: 512, label: 'v1' }], in_progress: [{ done: 100, of: 512 }],
    gate: { candidate: 'v1', reference: 'v0', reference_same_series: false, pair: null } };
  const waiting = progress({ held });
  assert.equal(waiting.headline, '63.7%');
  assert.match(waiting.detail, /Not paired yet.*100\/512/);
  const noise = progress({ held: { ...held, gate: { ...held.gate, pair: { delta_pts: 2.1, ci_pts: [-1.0, 5.2], tasks: 256 } } } });
  assert.equal(noise.tone, 'flat');
  assert.match(noise.detail, /not distinguishable from noise/);
  assert.equal(progress({ held: { ...held, gate: { ...held.gate, pair: { delta_pts: 4, ci_pts: [0.5, 7], tasks: 256 } } } }).tone, 'good');
  assert.equal(progress({ held: { ...held, gate: { ...held.gate, pair: { delta_pts: -4, ci_pts: [-7, -0.5], tasks: 256 } } } }).tone, 'bad');
  assert.equal(progress(null).headline, '—');
});

test('health lists exporter alerts, a held regression, stale data, and respects acknowledgements', () => {
  const snap = { ts: NOW - 10e3, alerts: [{ id: 'a', level: 'warn', text: 'warn' }, { id: 'b', level: 'crit', text: 'crit' }], runs: [],
    ei: { held: { gate: { candidate: 'v2', reference: 'v1', pair: { delta_pts: -5, ci_pts: [-8, -2] } } } } };
  const hv = health(snap, NOW, {});
  assert.equal(hv.level, 'crit');
  assert.equal(hv.items.length, 3);
  assert.equal(health(snap, NOW, { a: 1, b: 1, 'held:v2:v1': 1 }).level, 'ok');
  assert.equal(health({ ...snap, alerts: [], ei: null, ts: NOW - 400e3 }, NOW, {}).level, 'warn');
  assert.equal(health(null, NOW).title, 'Waiting for data');
});

test('activity groups what runs where and hides plumbing', () => {
  const snap = { ei: { rounds: [{ id: 'r0004', host: 'colab-g4', sft_running: true, sft: { step: 5, of: 10 } }], collection: { actors: [{ host: 'dgx', version: 3, rate_h: 270 }] } },
    processes: [{ state: 'running', role: 'held', machine: 'pop', title: 'Held eval', where: 'pop' }, { state: 'running', role: 'held', machine: 'dgx', title: 'Held eval', where: 'pop' },
      { state: 'running', role: 'datafill', machine: 'dgx', title: 'Vibestudio student data', where: 'dgx' }, { state: 'running', role: 'datafill', machine: 'dgx', title: 'Vibestudio student data', where: 'dgx' },
      { state: 'running', role: 'plumbing', machine: 'dgx', title: 'x', where: 'dgx' }, { state: 'failed', role: 'tuning', machine: 'dgx', title: 'y', where: 'colab' }] };
  const a = activity(snap);
  assert.deepEqual(a['colab-g4a'].map(x => x.text), ['SFT of r0004 · step 5/10']);
  assert.deepEqual(a.dgx.map(x => x.text), ['Collecting with v3 · 270/h', 'Vibestudio student data ×2']);
  assert.deepEqual(a.pop.map(x => x.text), ['Held eval']);
});

test('listener status combines the inbox worker presence with the supervising agent heartbeat', async () => {
  const { listenerStatus } = await import('../pages/ops/ui/model.mjs');
  const up = { status: 'available' }, tick = (over = {}) => ({ ts: NOW, at: NOW, listener: NOW - 20e3, ...over });
  assert.equal(listenerStatus(null).state, 'unknown');
  assert.equal(listenerStatus(tick(), NOW, null).state, 'none');                               // no worker card yet
  assert.equal(listenerStatus(tick(), NOW, { status: 'stale' }).state, 'none');
  assert.match(listenerStatus(tick(), NOW, { status: 'stopped' }).text, /ops-inbox/);
  assert.equal(listenerStatus(tick(), NOW, up).state, 'live');                                  // worker up and the agent read the inbox 20 s ago
  assert.equal(listenerStatus(tick(), NOW, { status: 'busy' }).state, 'live');
  const recorded = listenerStatus(tick({ listener: NOW - 600e3 }), NOW, up);
  assert.equal(recorded.state, 'recorded');
  assert.match(recorded.text, /last read the inbox 10m/);
  assert.equal(listenerStatus(tick({ listener: null }), NOW, up).state, 'recorded');            // the agent has never read it
  assert.equal(listenerStatus(tick({ at: NOW - 600e3 }), NOW, up).state, 'unknown');           // an old tick says nothing
});

test('askState follows a question from this device through the journal to the answer', async () => {
  const { askState, inboxSummary } = await import('../pages/ops/ui/model.mjs');
  const subject = { kind: 'alert', id: 'ei:gate', title: 't' };
  const pending = [{ id: 'op-1', name: 'ask', state: 'saved', ts: NOW, input: { kind: 'ask', subject, text: 'why?' } },
    { id: 'op-x', name: 'ask', state: 'saved', ts: NOW, input: { kind: 'ask', subject: { kind: 'unit', id: 'u' }, text: 'other' } }];
  let st = askState({ asks: [], answers: [] }, pending, {}, [], 'alert', 'ei:gate');
  assert.deepEqual([st.state, st.confirmed, st.at, st.text], ['waiting', false, NOW, 'why?']);
  assert.equal(askState({ asks: [], answers: [] }, pending, {}, [], 'alert', 'other-id'), null);
  const inbox = { asks: [{ operationId: 'op-1', kind: 'ask', subject, text: 'why?' }, { operationId: 'ack-1', kind: 'ack_alert', subject, text: 'ack' }], answers: [] };
  st = askState(inbox, [], { 'op-1': { at: NOW } }, [], 'alert', 'ei:gate');          // applied: the pending record is gone, the ask is in the journal; an acknowledgement is not a question
  assert.deepEqual([st.state, st.confirmed, st.at], ['waiting', true, NOW]);
  assert.deepEqual(inboxSummary(inbox, [], []), { waiting: 1, unread: 0, total: 1 });
  inbox.answers.push({ ask: 'op-1', text: 'because', operationId: 'ans-1' });
  st = askState(inbox, [], {}, [], 'alert', 'ei:gate');
  assert.deepEqual([st.state, st.answer, st.unread], ['answered', 'because', true]);
  assert.equal(askState(inbox, [], {}, ['ans-1'], 'alert', 'ei:gate').unread, false);
  assert.deepEqual(inboxSummary(inbox, [], []), { waiting: 0, unread: 1, total: 1 });
  assert.deepEqual(inboxSummary(inbox, [{ id: 'op-2', name: 'ask', input: { kind: 'ask' } }], ['ans-1']), { waiting: 1, unread: 0, total: 2 });
});

/* 2026-11-24: a reopened page showed "Exporter data is 4h 30m old" (crit) while the exporter wrote every 30 s; the page held an old
   snapshot and could not reach the DGX. Age alone must never raise the exporter alarm. */
test('a held snapshot after reopening is refreshing, not an exporter alarm', () => {
  const snap = { v: 3, ts: NOW - 4.5 * 3600e3 };
  const since = NOW - 5e3;
  for (const read of [{ since, cached: true }, { since, okAt: NOW - 5 * 3600e3 }, { since }]) {
    const hv = health(snap, NOW, {}, read);
    assert.equal(hv.feed.state, 'refreshing');
    assert.ok(!hv.items.some(i => i.id === 'stale' || i.id === 'link'), JSON.stringify(hv.items));
    assert.equal(hv.level, 'ok');
  }
});

test('a failed read names the missing connection, not the exporter', () => {
  const snap = { v: 3, ts: NOW - 4.5 * 3600e3 };
  const hv = health(snap, NOW, {}, { since: NOW - 60e3, okAt: NOW - 4.5 * 3600e3, failAt: NOW - 10e3, error: 'No answer to ops.snapshot from the DGX yet.' });
  assert.equal(hv.feed.state, 'link_down');
  const link = hv.items.find(i => i.id === 'link');
  assert.match(link.text, /No connection to the DGX/);
  assert.equal(link.level, 'warn');
  assert.ok(!hv.items.some(i => i.id === 'stale'));
  // a cached snapshot whose first read failed is also a connection problem
  assert.equal(health(snap, NOW, {}, { since: NOW - 60e3, failAt: NOW - 1e3, cached: true }).feed.state, 'link_down');
  // a failure while the data is still young is not an alert
  assert.equal(health({ v: 3, ts: NOW - 60e3 }, NOW, {}, { since: NOW - 90e3, okAt: NOW - 60e3, failAt: NOW - 1e3 }).items.length, 0);
});

test('a successful read of old data says the exporter is behind', () => {
  const snap = { v: 3, ts: NOW - 20 * 60e3 };
  const hv = health(snap, NOW, {}, { since: NOW - 120e3, okAt: NOW - 2e3 });
  assert.equal(hv.feed.state, 'exporter_behind');
  const st = hv.items.find(i => i.id === 'stale');
  assert.match(st.text, /^Exporter behind/);
  assert.equal(st.level, 'crit');
  assert.equal(health({ v: 3, ts: NOW - 200e3 }, NOW, {}, { since: NOW - 300e3, okAt: NOW - 2e3 }).items[0].level, 'warn');
  // a fresh read of fresh data is live, and a later failed read only matters once the data is old
  assert.equal(health({ v: 3, ts: NOW - 20e3 }, NOW, {}, { since: NOW - 300e3, okAt: NOW - 2e3 }).feed.state, 'live');
});

test('clock skew from a signal signing time', () => {
  assert.equal(clockSkew(NOW - 2400, NOW), 2400);
  assert.equal(clockSkew(undefined, NOW), null);
  assert.equal(SIGNAL_CLOCK_SKEW_MS, 2000);
});

test('progress reads the candidate against the fixed anchor, counts every sample, and names a cross-series pair', () => {
  const held = { results: [{ version: 12, label: 'v12', rate: 0.5742, successes: 294, counted: 512, unscored: 2, set: 'held-v3', scored: 510, ran: 512 }],
    no_gain: { text: '3 consecutive neutral rounds and no policy beats the anchor.' },
    gate: { candidate: 'v12', reference: 'v10', anchor: 'v0 (sft-v9)', candidate_result: { set: 'held-v3' },
      vs_anchor: { delta_pts: 0, ci_pts: [-4.4, 4.4], tasks: 256, same_series: false }, pair: { delta_pts: -1.2, ci_pts: [-5.3, 3], tasks: 256, same_series: false } } };
  const p = progress({ held });
  assert.match(p.headline, /0\.0 pts/);
  assert.match(p.sub, /v12 vs v0 \(sft-v9\) \(fixed anchor\), held-v3, across series/);
  assert.match(p.detail, /not distinguishable from noise\. 3 consecutive neutral rounds/);
  assert.equal(p.tone, 'flat');
  // no pair at all: the pass rate itself, with the unscored samples named
  const q = progress({ held: { results: held.results, gate: { candidate: 'v12', reference: 'v10', reference_same_series: false } } });
  assert.equal(q.headline, '57.4%');
  assert.match(q.sub, /294\/512 samples \(2 unscored count as failures\)/);
  // a regression against the anchor alarms even when the reference pair does not
  const hv = health({ v: 4, ts: NOW - 5e3, alerts: [], runs: [], ei: { held: { gate: { candidate: 'v13', reference: 'v10', anchor: 'v0', vs_anchor: { delta_pts: -6, ci_pts: [-10, -2] }, pair: { delta_pts: 1, ci_pts: [-3, 5] } } } } }, NOW, {});
  assert.deepEqual(hv.items.map(i => i.id), ['held:v13:v0']);
});
