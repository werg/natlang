import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { dumpNativeState, loadFunctionFile, NatlangHost } from '../dist/index.js';
import { WorkflowService } from '../../applications/workflow_service.mjs';

const path = fileURLToPath(new URL('../../codebases/api_workflow/step.nl', import.meta.url));

test('natlang Fold recovers a lost payment acknowledgement without a second charge', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-workflow-'));
  const service = new WorkflowService(root);
  const init = await service.open('order1', 1200);
  const host = new NatlangHost({ host: { workflow: service,
    drainEvents: () => service.drainEvents() } });
  const step = dumpNativeState(loadFunctionFile(path));
  const choices = ['reserve', 'charge', 'reconcile', 'ship'];
  try {
    const result = await host.run({ source: { kind: 'program', program: { $fold: {
      type: 'Fold<WorkflowEvent, WorkflowState>', types: step.$lambda.types,
      init, step, over: [{ kind: 'continue' }, { kind: 'continue', fault: 'lost_ack' },
        { kind: 'reconcile' }, { kind: 'continue' }] } } },
      options: { model: { segment_turns: 2 } },
      modelTurn: turn => {
        if (turn.messages.filter(m => m.role === 'assistant').length > 1)
          return { calls: [], text: 'done', completion_tokens: 1 };
        const prompt = String(turn.messages.find(m => m.role === 'user')?.content ?? '');
        if (prompt.includes('function step(')) return { calls: [
          ['call', { function: 'inspect', to: 'let/current', inputs: { order_id: 'args/acc/order_id' } }],
          ['call', { function: 'choose', to: 'let/decision', inputs: {
            current: 'let/current', item: 'args/item' } }],
          ['call', { function: 'apply', to: 'return', inputs: {
            current: 'let/current', item: 'args/item', decision: 'let/decision' } }],
        ], completion_tokens: 1 };
        return { calls: [['write', { path: 'return', value: {
          action: choices.shift(), reason: 'Follow durable workflow state' } }]], completion_tokens: 1 };
      } });
    assert.equal(result.outcome.kind, 'done');
    assert.equal(result.value.phase, 'shipped');
    assert.equal(result.value.pending, '');
    assert.deepEqual((await service.remoteEffects()).map(row => row.action),
      ['reserve', 'charge', 'ship']);
    const restarted = new WorkflowService(root);
    assert.equal((await restarted.read('order1')).phase, 'shipped');
    const replay = await restarted.apply('order1', 0, { kind: 'continue' },
      { action: 'charge', reason: 'stale' });
    assert.equal(replay.phase, 'shipped');
    assert.equal((await restarted.remoteEffects()).filter(row => row.action === 'charge').length, 1);
  } finally { host.close(); await rm(root, { recursive: true, force: true }); }
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
