import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime } from '../dist/index.js';
import { ScheduleWorkspace, plan } from '../../applications/dist/scheduling/index.js';
import { scriptedModel } from './support/natlang.mjs';

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

test('natlang ranks exact feasible schedules and the workspace conditionally commits', async () => {
  const scheduler = new ScheduleWorkspace(day);
  const model = scriptedModel(opening => {
    assert.match(opening, /request: string/);
    assert.match(opening, /alternatives: Alternatives|alternatives: \{/);
    return 'return alternatives.options[0]';
  });
  const result = await createNatlangRuntime({ model: model.driver }).run(() => plan(scheduler, 'Draft early, then review'));
  assert.equal(result.status, 'committed', result.detail);
  assert.equal(result.plan.length, 2);
  assert.ok(result.plan[1].start >= result.plan[0].end);
  assert.deepEqual(scheduler.drainEvents().map(event => event.operation), ['schedule.alternatives', 'schedule.commit']);
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
