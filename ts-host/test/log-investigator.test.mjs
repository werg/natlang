import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime } from '../dist/index.js';
import { LogWorkspace, LogIndex, commit, emptyIncidentState, step, defaultSettings } from '../../applications/dist/logs/index.js';
import { scriptedModel } from './support/natlang.mjs';

const line = (id, cursor, code, message, level = 'WARN', service = 'api') => ({ kind: 'log', id, cursor,
  occurred_at: cursor * 1000, arrived_at: cursor * 1000 + 100, service, code, level, message });
const gap = (id, cursor, message) => ({ kind: 'gap', id, cursor, occurred_at: cursor * 1000, arrived_at: cursor * 1000,
  service: 'api', code: '', level: '', message });

/** The investigation's stages, scripted as eval code a model might write for each stage's instructions. */
const STAGES = {
  investigate: `
    const IGNORE = { action: 'ignore', severity: '', claim: '', uncertainty: '', hypothesis_id: '', cited: [] };
    if (item.kind === 'gap') return { significance: 'ignore', incident: null, folded: [], escalation: IGNORE, effects: [], gap: await gap(item, incidents) };
    const s = await significance(item, observation, settings);
    if (s === 'ignore') return { significance: s, incident: null, folded: [], escalation: IGNORE, effects: [], gap: null };
    const open = incidents.filter(row => row.closes_at >= item.occurred_at);
    const a = await cluster(item, open);
    const union = (x, y) => [...new Set([...x, ...y])];
    let inc, fresh = false;
    if (a.action === 'open') inc = { id: item.id, opened_at: item.occurred_at, last_seen: item.occurred_at, closes_at: item.occurred_at + settings.quiet_ms,
      services: [item.service], codes: [item.code], members: [item.id], count: 1, severity: '', hypotheses: [], alert_key: null, summary: '' };
    else {
      const target = open.find(row => row.id === a.incident_id);
      fresh = !target.services.includes(item.service) || !target.codes.includes(item.code);
      inc = { ...target, last_seen: Math.max(target.last_seen, item.occurred_at), closes_at: Math.max(target.closes_at, item.occurred_at + settings.quiet_ms),
        services: union(target.services, [item.service]), codes: union(target.codes, [item.code]),
        members: [...target.members, item.id].slice(-settings.max_members), count: target.count + 1 };
    }
    for (const id of a.fold) {
      const other = open.find(row => row.id === id);
      inc = { ...inc, members: [...other.members, ...inc.members].slice(-settings.max_members), count: inc.count + other.count,
        services: union(inc.services, other.services), codes: union(inc.codes, other.codes), opened_at: Math.min(inc.opened_at, other.opened_at),
        hypotheses: [...inc.hypotheses, ...other.hypotheses.filter(h => !inc.hypotheses.some(x => x.id === h.id))], alert_key: inc.alert_key ?? other.alert_key };
    }
    if (inc.count < settings.threshold && s !== 'urgent') return { significance: s, incident: inc, folded: a.fold, escalation: IGNORE, effects: [], gap: null };
    const note = files ? await runbook(item, files) : '';
    if (!inc.hypotheses.length || fresh) inc = { ...inc, hypotheses: await hypothesize(inc, item, note) };
    const found = await gather(await queries(inc, inc.hypotheses, item, settings));
    const supports = await Promise.all(inc.hypotheses.map(h => weigh(h, found.filter(f => f.hypothesis_id === h.id))));
    const e = await escalate(inc, inc.hypotheses, supports, item, observation, settings);
    if (e.severity) inc = { ...inc, severity: e.severity };
    let effects = [];
    if (e.action === 'escalate') {
      inc = { ...inc, summary: await summarize(inc, inc.hypotheses, supports, e), alert_key: inc.id };
      effects = [{ kind: 'alert', key: inc.id, service: item.service, code: item.code, claim: e.claim, evidence_ids: e.cited }];
    }
    return { significance: s, incident: inc, folded: a.fold, escalation: e, effects, gap: null };`,
  cluster: `
    const ds = await Promise.all(incidents.map(row => decide(kin, item, row)));
    const matches = incidents.filter((row, i) => ds[i].value === 'same' && ds[i].confidence >= 0.5).sort((x, y) => x.opened_at - y.opened_at);
    if (!matches.length) return { action: 'open', incident_id: item.id, fold: [], reason: 'nothing open fits' };
    return { action: 'join', incident_id: matches[0].id, fold: matches.slice(1).map(row => row.id), reason: 'same service' };`,
  hypothesize: `return [{ id: 'h1', claim: 'The auth backend is failing', kind: 'cause' }, { id: 'h2', claim: 'A test is running', kind: 'benign' }];`,
  queries: `return hypotheses.map(h => ({ hypothesis_id: h.id, ...(h.kind === 'benign' ? { contains: 'test' } : { service: item.service, code: item.code }),
    from: item.occurred_at - settings.window_ms, to: item.occurred_at, limit: 20 }));`,
  weigh: `
    const evidence = found.flatMap(f => f.evidence);
    return { hypothesis_id: hypothesis.id, stance: evidence.length ? 'supports' : 'neutral', evidence_ids: evidence.map(e => e.id), note: evidence.length + ' records' };`,
  escalate: `
    const met = hypotheses.filter(h => h.kind !== 'benign').map(h => {
      const mine = supports.filter(s => s.hypothesis_id === h.id);
      return { h, cited: [...new Set(mine.filter(s => s.stance === 'supports').flatMap(s => s.evidence_ids))].sort(),
        against: new Set(mine.filter(s => s.stance === 'contradicts').flatMap(s => s.evidence_ids)).size };
    }).filter(row => row.cited.length >= settings.threshold && row.against < row.cited.length).sort((x, y) => y.cited.length - x.cited.length);
    const winner = met[0];
    const benign = winner && supports.some(s => hypotheses.find(h => h.id === s.hypothesis_id).kind === 'benign' && s.stance === 'supports' && winner.cited.every(id => s.evidence_ids.includes(id)));
    const won = winner && !benign;
    const action = won && incident.alert_key === null && !observation.late ? 'escalate' : 'investigate';
    return { action, severity: won ? await decide(severity, winner.h.claim, winner.cited.length, incident.services).then(d => d.value) : incident.severity,
      claim: won ? winner.h.claim + ' in ' + incident.services.join(', ') : '', uncertainty: '', hypothesis_id: won ? winner.h.id : '', cited: won ? winner.cited : [] };`,
  summarize: `return 'Repeated failures in ' + incident.services.join(', ') + ' (' + incident.count + ' events).';`,
  gap: `return { affected: incidents.filter(row => row.opened_at <= item.occurred_at && item.occurred_at <= row.closes_at).map(row => row.id), unknown: 'Source gap at cursor ' + item.cursor + ': ' + item.message };`,
};

function logModel(overrides = {}) {
  const seen = [];
  const model = scriptedModel(opening => {
    const stage = /inside this call: (\w+)\(/.exec(opening)?.[1];
    seen.push(stage);
    return overrides[stage] ?? STAGES[stage] ?? null;
  });
  const decide = async ({ messages, options }) => {
    const text = JSON.stringify(messages).replace(/\\"/g, '"').replace(/\\\\/g, '\\');
    const stage = /inside this call: (\w+)\(/.exec(text)?.[1];
    seen.push(stage);
    let winner;
    if (stage === 'judge') winner = /message: "[^"]*(?:test|probe ok)[^"]*"/.test(text) ? 'ignore' : /message: "[^"]*(?:failed|timed out)[^"]*"/.test(text) ? 'watch' : 'ignore';
    else if (stage === 'kin') winner = /service: "(\w+)"/.exec(text)?.[1] && text.includes('services: ["' + /service: "(\w+)"/.exec(text)[1] + '"') ? 'same' : 'different';
    else winner = 'high';
    return { log_probs: options.map(option => Math.log(option === JSON.stringify(winner) ? 0.9 : 0.1 / (options.length - 1))) };
  };
  return { driver: Object.assign(model.driver, { decide }), seen };
}
const count = (seen, name) => seen.filter(stage => stage === name).length;

test('natlang investigates a burst: significance, clustering, hypotheses over the index, escalation by the evidence rule, one alert', async () => {
  const sent = [];
  const logs = new LogWorkspace({ sendAlert: async alert => { sent.push(alert); return { status: 'sent', detail: 'accepted' }; } });
  const third = line('e3', 2, 'auth_failed', 'Third failed login');
  const events = [line('e1', 0, 'auth_failed', 'First failed login'), line('e2', 1, 'auth_failed', 'Second failed login'), third, { ...third },
    gap('gap1', 4, 'collector unavailable'), line('e5', 5, 'auth_failed', 'Fourth failed login'),
    line('e6', 6, 'health_probe', 'ERROR token used in test fixture')];
  const model = logModel();
  const runtime = createNatlangRuntime({ model: { driver: model.driver, maxTurns: 6 } });
  let state = emptyIncidentState();
  for (const event of events) state = await runtime.run(() => step(logs, state, event), logs.runOptions());
  assert.equal(state.observed, 5);
  assert.equal(state.alerts.length, 1, 'the fourth failure joins the incident and does not alert again');
  assert.equal(sent.length, 1);
  assert.deepEqual(sent[0].evidence_ids, ['e1', 'e2', 'e3']);
  assert.equal(state.incidents.length, 1);
  assert.equal(state.incidents[0].count, 4);
  assert.equal(state.incidents[0].alert_key, 'e1');
  assert.equal(state.incidents[0].severity, 'high');
  assert.match(state.incidents[0].summary, /Repeated failures in api/);
  assert.match(state.unknowns[0], /collector unavailable/);
  assert.equal(state.incidents[0].closes_at, 5000 + 300_000, 'the timer is state');
  assert.equal(count(model.seen, 'judge'), 5, 'every line is judged; duplicates and gaps are not');
  assert.equal(count(model.seen, 'investigate'), 6);
  assert.equal(count(model.seen, 'hypothesize'), 1, 'hypotheses are generated once for the incident');
  assert.equal(count(model.seen, 'escalate'), 2, 'e3 escalates, e5 only investigates');
  assert.ok(logs.drainEvents().some(event => event.operation === 'logs.query'), 'evidence was fetched from the exact index');
});

test('the crisp significance setting judges lines without the model; quiet incidents close on their timer', async () => {
  const logs = new LogWorkspace({ significance: 'crisp', threshold: 99, quietMs: 10_000 });
  const model = logModel();
  const runtime = createNatlangRuntime({ model: { driver: model.driver, maxTurns: 6 } });
  let state = emptyIncidentState();
  for (const event of [line('i1', 0, 'note', 'all fine', 'INFO'), line('w1', 1, 'slow', 'slow response', 'WARN'),
    line('w2', 2, 'slow', 'slow response again', 'WARN')]) state = await runtime.run(() => step(logs, state, event), logs.runOptions());
  assert.equal(count(model.seen, 'judge'), 0);
  assert.equal(state.incidents.length, 1);
  assert.equal(state.status, 'observing');
  state = await runtime.run(() => step(logs, state, line('w3', 60, 'other', 'timed out', 'WARN', 'db')), logs.runOptions());
  assert.deepEqual(state.closed.map(row => row.id), ['w1'], 'the first incident closed at its timer');
  assert.deepEqual(state.incidents.map(row => row.id), ['w3']);
});

test('the index answers exact windowed queries; the sink demands real evidence and is idempotent and honest about unknown delivery', async () => {
  const index = new LogIndex(60_000);
  const items = [line('a', 0, 'failure', 'one'), line('b', 1, 'failure', 'two timed out'), line('c', 2, 'failure', 'three', 'ERROR', 'db')];
  items.forEach(item => index.observe(item));
  assert.equal(index.observe(items[0]).status, 'duplicate');
  assert.throws(() => index.observe({ ...items[0], message: 'changed' }), /reused/);
  assert.deepEqual(index.search({ service: 'api', from: 0, to: 5000, limit: 1 }), { total: 2, evidence: [{ id: 'a', service: 'api', code: 'failure', occurred_at: 0, level: 'WARN', message: 'one' }] });
  assert.deepEqual(index.search({ contains: 'TIMED', from: 0, to: 5000, limit: 9 }).evidence.map(row => row.id), ['b']);
  assert.deepEqual(index.search({ level: 'error', from: 0, to: 5000, limit: 9 }).evidence.map(row => row.id), ['c']);

  const logs = new LogWorkspace({ sendAlert: async () => { throw new Error('network outcome unknown'); } });
  items.forEach(item => logs.index.observe(item));
  const effect = { kind: 'alert', key: 'k', service: 'api', code: 'failure', claim: 'Failures', evidence_ids: ['a', 'b', 'c'] };
  assert.equal((await logs.sink.deliver({ ...effect, evidence_ids: ['a', 'invented'] })).status, 'insufficient');
  assert.equal((await logs.sink.deliver({ ...effect, evidence_ids: [] })).status, 'insufficient');
  assert.equal((await logs.sink.deliver(effect)).status, 'unknown');
  assert.equal((await logs.sink.deliver(effect)).status, 'duplicate');
});

test('commit folds incidents, retires closed ones, lets a refused alert be retried and records gaps', () => {
  const incident = (id, extra = {}) => ({ id, opened_at: 0, last_seen: 0, closes_at: 100_000, services: ['api'], codes: ['x'], members: [id], count: 1,
    severity: '', hypotheses: [], alert_key: null, summary: '', ...extra });
  const none = { action: 'ignore', severity: '', claim: '', uncertainty: '', hypothesis_id: '', cited: [] };
  const state = { ...emptyIncidentState(), incidents: [incident('a'), incident('b'), incident('old', { closes_at: 10 })] };
  const merged = commit(state, line('e', 5, 'x', 'm'), { significance: 'watch', incident: incident('a', { count: 3 }), folded: ['b'], escalation: none, effects: [], gap: null }, []);
  assert.deepEqual(merged.incidents.map(row => [row.id, row.count]), [['a', 3]]);
  assert.deepEqual(merged.closed.map(row => row.id), ['old']);
  assert.equal(merged.status, 'observing');
  const refused = commit(state, line('f', 6, 'x', 'm'), { significance: 'watch', incident: incident('a', { alert_key: 'a' }), folded: [],
    escalation: { ...none, action: 'escalate' }, effects: [{}], gap: null }, [{ status: 'insufficient', key: 'a', detail: '' }]);
  assert.equal(refused.incidents.find(row => row.id === 'a').alert_key, null);
  assert.equal(refused.alerts.length, 0);
  assert.equal(refused.status, 'investigating');
  const gapped = commit(refused, gap('g', 7, 'collector down'), { significance: 'ignore', incident: null, folded: [], escalation: none, effects: [], gap: { affected: ['a'], unknown: 'Source gap at cursor 7: collector down' } }, []);
  assert.equal(gapped.status, 'gap');
  assert.equal(gapped.incidents.find(row => row.id === 'a').closes_at, 7000 + defaultSettings.quiet_ms);
  assert.match(gapped.unknowns[0], /collector down/);
});
