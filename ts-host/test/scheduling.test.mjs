import assert from 'node:assert/strict';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { NatlangHost } from '../dist/index.js';
import { ScheduleWorkspace } from '../../applications/scheduling.mjs';

const path = fileURLToPath(new URL('../../codebases/scheduler/plan.nl', import.meta.url));
const day = {
  windows: [{ start: '2026-09-21T09:00:00+02:00', end: '2026-09-21T12:00:00+02:00' }],
  fixed: [{ id: 'meeting', start: '2026-09-21T10:00:00+02:00',
    end: '2026-09-21T10:30:00+02:00' }],
  tasks: [
    { id: 'draft', minutes: 30, earliest: '2026-09-21T09:00:00+02:00',
      latest: '2026-09-21T11:30:00+02:00', after: [] },
    { id: 'review', minutes: 30, earliest: '2026-09-21T09:00:00+02:00',
      latest: '2026-09-21T12:00:00+02:00', after: ['draft'] },
  ],
};

test('natlang ranks exact feasible schedules and host conditionally commits', async () => {
  const scheduler = new ScheduleWorkspace(day);
  const option = scheduler.alternatives().options[0];
  const host = new NatlangHost({ host: { scheduler,
    drainEvents: () => scheduler.drainEvents() } });
  try {
    const result = await host.run({ source: { kind: 'file', path },
      inputs: { request: 'Draft early, then review' },
      options: { model: { segment_turns: 2 } },
      modelTurn: turn => {
        if (turn.messages.filter(m => m.role === 'assistant').length > 1)
          return { calls: [], text: 'done', completion_tokens: 1 };
        const prompt = String(turn.messages.find(m => m.role === 'user')?.content ?? '');
        if (prompt.includes('function plan(')) return { calls: [
          ['call', { function: 'inspect', to: 'let/snapshot' }],
          ['call', { function: 'enumerate', to: 'let/alternatives' }],
          ['call', { function: 'rank', to: 'let/chosen', inputs: {
            request: 'args/request', snapshot: 'let/snapshot',
            alternatives: 'let/alternatives' } }],
          ['call', { function: 'commit', to: 'return', inputs: {
            chosen: 'let/chosen', revision: 'let/snapshot/revision' } }],
        ], completion_tokens: 1 };
        return { calls: [['write', { path: 'return', value: option }]], completion_tokens: 1 };
      } });
    assert.equal(result.outcome.kind, 'done');
    assert.equal(result.value.status, 'committed');
    assert.equal(result.value.plan.length, 2);
    assert.ok(result.value.plan[1].start >= result.value.plan[0].end);
  } finally { host.close(); }
});

test('new observations invalidate stale proposals and impossible schedules remain explicit', () => {
  const scheduler = new ScheduleWorkspace(day);
  const initial = scheduler.alternatives();
  const candidate = initial.options[0];
  const block = { kind: 'block', id: 'urgent',
    start: new Date(candidate.slots[0].start * 60000).toISOString(),
    end: new Date(candidate.slots[0].end * 60000).toISOString() };
  scheduler.observe(block);
  assert.equal(scheduler.commit(candidate, initial.revision).status, 'stale');
  assert.equal(scheduler.commit(candidate, scheduler.revision).status, 'rejected');
  assert.equal(scheduler.observe(block).revision, 1);
  assert.throws(() => scheduler.observe({ ...block, end: '2026-09-21T12:00:00+02:00' }), /changed event ID/);
  assert.throws(() => new ScheduleWorkspace({ ...day,
    tasks: day.tasks.map(task => ({ ...task, after: ['review'] })) }), /dependency cycle/);
  assert.throws(() => new ScheduleWorkspace({ ...day,
    windows: [{ start: '2026-09-21T09:00:00', end: '2026-09-21T12:00:00' }] }),
  /explicit UTC offset/);
});
