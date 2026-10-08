import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { CallStore } from '../dist/calls/store.js';
import { hostCaptures } from '../dist/calls/host-capture.js';
import { interpreter, lambda } from './support/natlang.mjs';

// The teacher collector drives a standalone native run with exact host capture (src/teacher/collector.ts). The run is
// recorded in the call store like any call, and the exact captures are a filter over that record.
test('a standalone run is recorded in the store, and its exact host captures can be rebuilt from the record', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-teacher-store-'));
  const store = CallStore.open(root);
  try {
    const capture = { definitionSources: ['price.nl'], inputArguments: ['id', 'absent'], captureOutput: true, captureAllOutputs: true, maxBytes: 1_048_576 };
    const runtime = interpreter({ calls: store, runId: 'teacher-run-1', modelId: 'scripted-teacher', manifest: { definition_source: 'price.nl' },
      services: { orders: { lookup: id => ({ id, total: id.length * 10, lines: [{ sku: 'a', qty: 2 }] }) } },
      exactHostTraceCapture: capture,
      agent: async session => { await session.applyAsync('eval', { code: 'const order = orders.lookup(id);\nreturn { total: order.total, lines: order.lines }' }); } });
    const result = await runtime.run(lambda({ type: '(id: string) => { total: number; lines: { sku: string; qty: number }[] }',
      instructions: 'Look up order id and return its total and lines.', args: { id: 'abc' } }));
    assert.equal(result.outcome.kind, 'done');

    const record = store.call('teacher-run-1');
    assert.ok(record, 'the root run is in the store under its run ID');
    assert.equal(record.definition.source, 'price.nl');
    assert.equal(record.executor.model_id, 'scripted-teacher');
    assert.equal(store.value(record.inputs.id), 'abc');
    assert.deepEqual(record.effects.map(effect => `${effect.service}.${effect.method}`), ['orders.lookup']);
    assert.deepEqual(store.value(record.effects[0].args), ['abc']);

    const fields = ['capture_kind', 'call_id', 'definition_source', 'name', 'complete', 'value', 'bytes', 'reason', 'value_sha256',
      'result_type', 'terminal_action_seq', 'origin'];
    const pick = item => Object.fromEntries(fields.filter(key => item[key] !== undefined).map(key => [key, item[key]]));
    const inline = runtime.trace.events.filter(event => event.kind === 'host_capture').map(pick);
    const rebuilt = hostCaptures(store, 'teacher-run-1', { inputArguments: capture.inputArguments, output: true, maxBytes: capture.maxBytes }).map(pick);
    assert.equal(inline.length, 3);
    assert.deepEqual(rebuilt, inline);
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});
