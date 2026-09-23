import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { dumpNativeState, loadFunctionFile, NatlangHost } from '../dist/index.js';
import { LogWorkspace } from '../../applications/log_investigator.mjs';
import { evalTurn } from './support/eval-turn.mjs';

const path = fileURLToPath(new URL('../../codebases/log_investigator/step.nl', import.meta.url));
const line = (id, cursor, code, message) => ({ kind: 'log', id, cursor,
  occurred_at: cursor * 1000, arrived_at: cursor * 1000 + 100,
  service: 'api', code, level: 'WARN', message });

test('natlang Fold investigates a burst, suppresses duplicate escalation and records a gap', async () => {
  const folder = mkdtempSync(join(tmpdir(), 'natlang-logs-'));
  const sent = [];
  const logs = new LogWorkspace({ sendAlert: async alert => {
    sent.push(alert); return { status: 'sent', detail: 'accepted' };
  } });
  const host = new NatlangHost({ host: { logs, drainEvents: () => logs.drainEvents() } });
  const step = dumpNativeState(loadFunctionFile(path));
  const init = { cursor: -1, observed: 0, alerts: [], unknowns: [], status: 'idle' };
  const third = line('e3', 2, 'auth_failed', 'Third failed login');
  const events = [line('e1', 0, 'auth_failed', 'First failed login'),
    line('e2', 1, 'auth_failed', 'Second failed login'), third,
    { ...third },
    { kind: 'gap', id: 'gap1', cursor: 4, occurred_at: 4000, arrived_at: 4000,
      service: 'api', code: '', level: '', message: 'collector unavailable' },
    line('e5', 5, 'health_probe', 'ERROR token used in test fixture'),
  ];
  let stepCount = 0, assessmentCount = 0;
  const tracePath = join(folder, 'run.trace.jsonl');
  try {
    const result = await host.run({ source: { kind: 'program', program: { $fold: {
      type: 'Fold<LogEvent, IncidentState>', types: step.$lambda.types, init, step,
      over: events } } }, tracePath, validationFeedback: 'caller',
      modelTurn: turn => {
        const prompt = String(turn.messages.find(m => m.role === 'user')?.content ?? '');
        if (prompt.includes('function step(')) {
          const index = stepCount++;
          if (index === 4) return evalTurn(turn, 'await gap(acc, item)');
          return evalTurn(turn,
            'const observation = await observe(item); const evidence = await search(observation); const judgement = await assess(item, observation, evidence, files); await transition(acc, item, observation, evidence, judgement)');
        }
        const index = assessmentCount++;
        return evalTurn(turn, `(${JSON.stringify({
          action: index === 2 || index === 3 ? 'escalate' : 'ignore', severity: 'medium',
          claim: 'Repeated failed logins', uncertainty: '',
        })})`);
      } });
    assert.equal(result.outcome.kind, 'done', JSON.stringify(result.outcome));
    assert.equal(result.value.observed, 4);
    assert.equal(result.value.alerts.length, 1);
    assert.equal(sent.length, 1);
    assert.match(result.value.unknowns[0], /collector unavailable/);
    const trace = readFileSync(tracePath, 'utf8').trim().split('\n').map(JSON.parse);
    assert.equal(trace.filter(e => e.kind === 'host' && e.event?.operation === 'logs.alert').length, 1);
  } finally { host.close(); rmSync(folder, { recursive: true, force: true }); }
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
