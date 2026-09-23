import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime } from '../dist/index.js';
import { LogWorkspace, emptyIncidentState, step } from '../../applications/dist/logs/index.js';
import { scriptedModel } from './support/natlang.mjs';

const line = (id, cursor, code, message) => ({ kind: 'log', id, cursor,
  occurred_at: cursor * 1000, arrived_at: cursor * 1000 + 100,
  service: 'api', code, level: 'WARN', message });

test('natlang investigates a burst, suppresses duplicate escalation and records a gap', async () => {
  const sent = [];
  const logs = new LogWorkspace({ sendAlert: async alert => { sent.push(alert); return { status: 'sent', detail: 'accepted' }; } });
  const third = line('e3', 2, 'auth_failed', 'Third failed login');
  const events = [line('e1', 0, 'auth_failed', 'First failed login'), line('e2', 1, 'auth_failed', 'Second failed login'), third,
    { ...third },
    { kind: 'gap', id: 'gap1', cursor: 4, occurred_at: 4000, arrived_at: 4000, service: 'api', code: '', level: '', message: 'collector unavailable' },
    line('e5', 5, 'auth_failed', 'Fourth failed login'),
    line('e6', 6, 'health_probe', 'ERROR token used in test fixture')];
  const model = scriptedModel(opening => {
    assert.match(opening, /Judge this event using the supplied exact log evidence/);
    return 'result = { action: evidence.length >= 3 && item.code === "auth_failed" ? "escalate" : "ignore", ' +
      'severity: "medium", claim: "Repeated failed logins", uncertainty: "" }';
  });
  const traces = [];
  const runtime = createNatlangRuntime({ model: model.driver, trace: trace => traces.push(trace) });
  let state = emptyIncidentState();
  for (const event of events) state = await runtime.run(() => step(logs, state, event));
  assert.equal(state.observed, 5);
  assert.equal(state.alerts.length, 1, 'the fourth failure joins the active incident');
  assert.equal(sent.length, 1);
  assert.match(state.unknowns[0], /collector unavailable/);
  assert.equal(model.openings.length, 5, 'duplicates and gaps are handled without the model');
  assert.ok(traces.every(trace => trace.outcome === 'done'));
});

test('alert sink rejects fabricated evidence and treats delivery errors as unknown', async () => {
  const logs = new LogWorkspace({ sendAlert: async () => { throw new Error('network outcome unknown'); } });
  const items = [line('a', 0, 'failure', 'one'), line('b', 1, 'failure', 'two'),
    line('c', 2, 'failure', 'three')];
  for (const item of items) logs.observe(item);
  const observation = logs.summary(items[2], 'new');
  const evidence = logs.query(observation);
  const judgement = { action: 'escalate', severity: 'high', claim: 'Three failures', uncertainty: '' };
  assert.equal((await logs.alert(items[2], observation,
    [{ ...evidence[0], message: 'invented' }, ...evidence.slice(1)], judgement)).status, 'insufficient');
  assert.equal((await logs.alert(items[2], observation, evidence, judgement)).status, 'unknown');
  assert.equal((await logs.alert(items[2], observation, evidence, judgement)).status, 'duplicate');
});
