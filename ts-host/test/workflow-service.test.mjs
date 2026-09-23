import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createNatlangRuntime } from '../dist/index.js';
import { WorkflowService, step } from '../../applications/dist/workflow/index.js';
import { scriptedModel } from './support/natlang.mjs';

test('natlang recovers a lost payment acknowledgement without a second charge', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-workflow-'));
  const service = new WorkflowService(root);
  await service.open('order1', 1200);
  const model = scriptedModel(() => 'const next = current.pending ? (event.kind === "reconcile" ? "reconcile" : "wait") : ' +
    '({ new: "reserve", reserved: "charge", charged: "ship" })[current.phase] ?? "wait";\n' +
    'result = { action: next, reason: "Follow durable workflow state" }');
  const runtime = createNatlangRuntime({ model: model.driver });
  try {
    let state;
    for (const event of [{ kind: 'continue' }, { kind: 'continue', fault: 'lost_ack' }, { kind: 'reconcile' }, { kind: 'continue' }])
      state = await runtime.run(() => step(service, 'order1', event));
    assert.equal(state.phase, 'shipped');
    assert.equal(state.pending, '');
    assert.deepEqual((await service.remoteEffects()).map(row => row.action), ['reserve', 'charge', 'ship']);
    const restarted = new WorkflowService(root);
    assert.equal((await restarted.read('order1')).phase, 'shipped');
    const replay = await restarted.apply('order1', 0, { kind: 'continue' }, { action: 'charge', reason: 'stale' });
    assert.equal(replay.phase, 'shipped');
    assert.equal((await restarted.remoteEffects()).filter(row => row.action === 'charge').length, 1);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('restart sees unknown charge, reconciles, and records failed compensation', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-workflow-'));
  try {
    let service = new WorkflowService(root);
    let state = await service.open('order2', 700);
    state = await service.apply('order2', state.revision, { kind: 'continue' },
      { action: 'reserve', reason: '' });
    state = await service.apply('order2', state.revision,
      { kind: 'continue', fault: 'lost_ack' }, { action: 'charge', reason: '' });
    assert.equal(state.pending, 'order2:charge');
    service = new WorkflowService(root);
    assert.equal((await service.read('order2')).pending, 'order2:charge');
    await assert.rejects(service.apply('order2', state.revision,
      { kind: 'continue' }, { action: 'charge', reason: '' }), /reconciled/);
    state = await service.apply('order2', state.revision,
      { kind: 'reconcile' }, { action: 'reconcile', reason: '' });
    assert.equal(state.phase, 'charged');
    state = await service.apply('order2', state.revision,
      { kind: 'continue', fault: 'definite_failure' }, { action: 'ship', reason: '' });
    assert.equal(state.phase, 'shipping-failed');
    state = await service.apply('order2', state.revision,
      { kind: 'continue', fault: 'rate_limit' }, { action: 'refund', reason: '' });
    assert.match(state.obligations[0], /compensation failed/);
    assert.equal((await service.remoteEffects()).filter(row => row.action === 'charge').length, 1);
  } finally { await rm(root, { recursive: true, force: true }); }
});
